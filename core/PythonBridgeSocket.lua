------------------------------------------------------------------------------------------
-- PythonBridgeSocket.lua - MAME LUA SCRIPT (CLIENT)
------------------------------------------------------------------------------------------
--
-- DESCRIPTION:
--   Ce script agit comme un pont de communication entre MAME et un agent externe en Python.
--   Il utilise le socket natif de MAME (`emu.file`) pour se connecter à un serveur Python,
--   éliminant ainsi le besoin de DLL externes comme `luasocket.dll`.
--
-- MODE DE FONCTIONNEMENT : CLIENT
--   - Lua (MAME) est le CLIENT.
--   - Python est le SERVEUR.
--   - Le script Python doit être lancé AVANT MAME pour que le port soit ouvert.
--
-- PROTOCOLE DE COMMUNICATION :
--   - Les messages sont envoyés par lots, terminés par une ligne "__end__".
--   - Les données sont compressées avec zlib pour optimiser la latence.
--   - Commandes supportées :
--     - "execute <commande>(<valeur>)" : Exécute une action sur un port d'entrée (IOPort).
--     - "read_memory <adresse>" : Lit un octet à une adresse mémoire.
--     - "read_memory_range <adresse>(<longueur>)" : Lit une plage d'octets.
--     - "write_memory <adresse>(<valeur>)" : Écrit un octet à une adresse.
--     - "wait_for <nombre>" : Mécanisme de synchronisation pour l'agent RL.
--
-- DOCUMENTATION UTILE :
--   - API Lua pour MAME : https://docs.mamedev.org/luascript/index.html
--   - Projet similaire (Tempest AI) : https://github.com/davepl/tempest_ai/
--
------------------------------------------------------------------------------------------

local debug = false
local flag = 0
local flag_wait_for = false
local frame = 0
local latence = 0.0
local port_game = 12345

-- Variables pour le buffer de lecture
local socket_buffer = ""
local command_queue = {}

-- Récupère le nom du "romset" en cours d'émulation
local game = emu.romname()

-- Références utiles pour MAME
local machine = manager.machine
local screen  = machine.screens[":screen"]
local cpu     = machine.devices[":maincpu"]
local mem     = cpu.spaces["program"]
local ioport  = machine.ioport

--------------------------------------------------------------------------------
-- Pour afficher les champs d'IOPort (debug éventuel)
--------------------------------------------------------------------------------
local function show_mame_game_ioports()
    for f, i in pairs(ioport.ports) do
        print(f, " - ioport:", i)
        for ff, j in pairs(ioport.ports[f].fields) do
            print("\tfield:", ff)
        end
    end
end

if debug then
    print(game)
    show_mame_game_ioports()
end

--------------------------------------------------------------------------------
-- Configuration du port en fonction du jeu et affectation des champs d'entrée
--------------------------------------------------------------------------------
local jeu
if game == "invaders" then
    port_game = 12345
    in0 = ioport.ports[":CONTP1"]
    in1 = ioport.ports[":IN1"]
    jeu = {
        P1_left     = in0.fields["P1 Left"],
        P1_right    = in0.fields["P1 Right"],
        P1_Button_1 = in0.fields["P1 Button 1"],
        P1_start    = in1.fields["1 Player Start"]
    }

elseif game == "pacman" then
    port_game = 12346
    dsw1 = ioport.ports[":DSW1"]
    in0  = ioport.ports[":IN0"]
    in1  = ioport.ports[":IN1"]
    jeu = {
        Lives     = dsw1.fields["Lives"],
        P1_Left   = in0.fields["P1 Left"],
        P1_Right  = in0.fields["P1 Right"],
        P1_Down   = in0.fields["P1 Down"],
        P1_Up     = in0.fields["P1 Up"],
        Coin_1    = in0.fields["Coin 1"],
        Coin_2    = in0.fields["Coin 2"],
        P1_start  = in1.fields["1 Player Start"]
    }
else
    -- Gérer d'autres jeux si nécessaire
end

for key, value in pairs(jeu) do
    print(key, value)
end

--------------------------------------------------------------------------------
-- OVERRIDE PORT VIA VARIABLE D'ENVIRONNEMENT
--------------------------------------------------------------------------------
local env_port = os.getenv("MAME_SOCKET_PORT")
if env_port then
    port_game = tonumber(env_port)
    print("Port surchargé par variable d'environnement MAME_SOCKET_PORT: " .. port_game)
end

--------------------------------------------------------------------------------
-- Gestion du socket serveur
--------------------------------------------------------------------------------
local host   = "127.0.0.1"
local port   = port_game

-- Créer l'objet fichier MAME (Native Socket)
-- 3 = OPEN_FLAG_READ | OPEN_FLAG_WRITE (Pas de CREATE pour un client, évite les conflits)
local client = emu.file(3)

print("MODE: MAME Native Socket (emu.file) - CLIENT")
print("Tentative de connexion au serveur Python sur socket." .. host .. ":" .. port)

local connected = false
-- Boucle de tentative de connexion (30 secondes max)
for i=1, 150 do 
    local err = client:open("socket." .. host .. ":" .. port)
    if not err then
        connected = true
        print("✅ Connecté au serveur Python !")
        break
    end
    -- Petite attente active (0.2s)
    local t = os.clock(); while os.clock() - t <= 0.2 do end
end

if not connected then
    print("❌ ECHEC connexion au serveur Python. Vérifiez que le script Python est lancé en premier.")
    client = nil
end

--------------------------------------------------------------------------------
-- Modules externes pour compression
--------------------------------------------------------------------------------
local base64 = require("base64")
local zlib   = require("zlib")

--------------------------------------------------------------------------------
-- Envoi de messages compressés vers Python
--------------------------------------------------------------------------------
local function send_to_python(messages)
    local message_str = table.concat(messages, "\n")
    local compressed_data, zlib_err = zlib.deflate()(message_str, "finish")
    if not compressed_data then
        print("Erreur ZLIB compression: " .. tostring(zlib_err))
        return
    end

    local status, err
    -- emu.file:write retourne (octets écrits) ou (nil, erreur)
    local bytes_written, write_err = client:write(compressed_data)
    if not bytes_written then
        status = false
        err = write_err
    else
        status = true
    end

    if not status then
        print("Erreur socket envoi: " .. tostring(err))
        if client then client:close() end
        client = nil
    end
end

--------------------------------------------------------------------------------
-- Réception des messages (Lecture bufferisée)
--------------------------------------------------------------------------------
local function receive_from_python()
    if not client then return {} end

    -- 1. Lire les données disponibles par gros blocs (4KB)
    -- emu.file:read() retourne ce qui est dispo ou nil/vide
    local chunk = client:read(4096)
    
    -- Si chunk est nil ou vide, pas de nouvelles données
    if not chunk or #chunk == 0 then
        return {}
    end

    socket_buffer = socket_buffer .. chunk

    -- 2. Extraire les lignes complètes du buffer
    while true do
        local start_pos, end_pos = socket_buffer:find("\n")
        if not start_pos then
            break -- Pas de ligne complète, on attend la suite
        end

        local line = socket_buffer:sub(1, start_pos - 1)
        if line:sub(-1) == "\r" then line = line:sub(1, -2) end

        socket_buffer = socket_buffer:sub(end_pos + 1)

        if line == "__end__" then
            -- On a un batch complet, on le renvoie et on vide la file d'attente
            local batch = command_queue
            command_queue = {}
            return batch
        elseif line ~= "" then
            table.insert(command_queue, line)
        end
    end

    -- Pas de batch complet ("__end__") trouvé, on retourne une table vide
    return {}
end

--------------------------------------------------------------------------------
-- Exécution de commande "execute <command>(<value>)"
--------------------------------------------------------------------------------
local function execute_command(command, value)
    -- Commandes communes
    if command == "throttle_rate" then
        machine.video.throttle_rate = tonumber(value)
    elseif command == "throttled" then
        machine.video.throttled = tonumber(value)
    else
        -- Commande liée au jeu
        local variable = rawget(jeu, command)
        if variable == nil then
            print(command, value)
            send_to_python({ "ERR: execute_command:" .. command .. " non comprise!" })
        else
            variable:set_value(tonumber(value))
        end
    end
end

--------------------------------------------------------------------------------
-- Lecture mémoire rapide (optimisée par blocs de 8 octets)
--------------------------------------------------------------------------------
local function read_memory_fast(start_addr_hex, length)
    local start = tonumber(start_addr_hex, 16)
    local len   = tonumber(length)
    local values = {}

    -- Lire par blocs de 8 octets
    for i = 0, len - 8, 8 do
        local addr = start + i
        local low  = mem:read_u32(addr) or 0
        local high = mem:read_u32(addr + 4) or 0

        for shift = 0, 24, 8 do
            table.insert(values, tostring((low >> shift) & 0xFF))
        end
        for shift = 0, 24, 8 do
            table.insert(values, tostring((high >> shift) & 0xFF))
        end
    end

    -- Terminer les derniers octets si len pas multiple de 8
    local remainder = len % 8
    if remainder > 0 then
        local addr = start + len - remainder
        for j = 0, remainder - 1 do
            local byte_value = mem:read_u8(addr + j) or 0
            table.insert(values, tostring(byte_value))
        end
    end

    return table.concat(values, ",")
end

--------------------------------------------------------------------------------
-- Lecture de mémoire simple (octet par octet)
--------------------------------------------------------------------------------
local function read_memory_range_func(start_addr_hex, length)
    local start = tonumber(start_addr_hex, 16)
    local len   = tonumber(length)
    local values = {}
    for i = 0, len - 1 do
        table.insert(values, tostring(mem:read_u8(start + i)))
    end
    return table.concat(values, ",")
end

--------------------------------------------------------------------------------
-- Lecture/Écriture d'un octet
--------------------------------------------------------------------------------
local function read_memory_value(address)
    return mem:read_u8(address)
end

local function write_memory_value(address, value)
    return mem:write_u8(address, value)
end

--------------------------------------------------------------------------------
-- Variables globales pour la gestion d'événements/messages
--------------------------------------------------------------------------------
local nb_messages_total_in_game = 0
local frame_in_game            = 0
local wait_for_messages        = 0
local frame_per_step           = 1
local x, y, _str               = 40, 10, ""

--------------------------------------------------------------------------------
-- Affichage à l'écran
--------------------------------------------------------------------------------
local function draw_on_screen()
    -- On vérifie que x et y ne sont pas nil ET qu'ils sont convertibles en nombres
    local posX = tonumber(x)
    local posY = tonumber(y)

    if posX and posY and _str then
        screen:draw_text(posX, posY, _str)
    end
end

--------------------------------------------------------------------------------
-- Fonction main() appelée chaque frame MAME
--------------------------------------------------------------------------------
local function main()
    if not client then return end -- Si pas de client, on ne fait rien

    frame = frame + 1
    -- if debug then print("frame:", frame) end
    -- Affichage de stats toutes les 100000 frames
    if frame % 100000 == 0 then
        print(
            "[" .. os.date() .. "]" ..
            "all frames:" .. frame .. " - frames_in_game:" .. frame_in_game ..
            " - " .. tostring(nb_messages_total_in_game) .. " messages traités => " ..
            tostring(frame_in_game > 0 and (nb_messages_total_in_game * 100 // frame_in_game / 100) or 0) .. " asks/frame_in_game " ..
            "[" .. tostring(nb_messages_total_in_game * 100 // frame / 100) .. "/frame]" ..
            " (frame_per_step=" .. tostring(frame_per_step) .. ")"
        )
    end

    -- On ne traite les messages que toutes les frame_per_step frames
    if frame % frame_per_step == 0 then
        local messages_in_game = 0

        while true do
            local messages_from_python = receive_from_python()
            local responses = {}

            if #messages_from_python > 0 then
                -- Parcours des messages reçus
                if debug then print("messages_from_python:", #messages_from_python) end
                for _, message_from_python in ipairs(messages_from_python) do

                    if message_from_python:find("execute") then
                        local command, value = message_from_python:match("execute ([%w_]+)%(([%d.]+)%)")
                        execute_command(command, tonumber(value))
                        table.insert(responses, "Command " .. message_from_python .. " ack")

                    elseif message_from_python:find("read_memory_range") then
                        local start_addr, length = message_from_python:match("read_memory_range (%x+)%((%d+)%)")
                        if start_addr and length then
                            local data = read_memory_fast(start_addr, length)
                            if debug then
                                print("read_memory_range: start=" .. start_addr .. " length=" .. length ..
                                      " -> #" .. #data)
                            end
                            table.insert(responses, data)
                        else
                            table.insert(responses, "ERR: format de read_memory_range incorrect")
                        end

                    elseif message_from_python:find("read_memory") then
                        local address = message_from_python:match("read_memory (%x+)")
                        local memory_value = read_memory_value(tonumber(address, 16))
                        table.insert(responses, tostring(memory_value))

                    elseif message_from_python:find("write_memory") then
                        local address, value = message_from_python:match("write_memory ([%w_]+)%((%d+)%)")
                        local memory_value   = write_memory_value(tonumber(address, 16), tonumber(value))
                        table.insert(responses, "W(" .. address .. ")=" .. value .. " (ACK)")

                    elseif message_from_python:find("draw_text") then
                        x, y, _str = message_from_python:match("draw_text%((%d+),(%d+),\"([^\"]+)\"%)")
                        table.insert(responses, "draw_text command (ACK)")
                    -- wait_for : Attend un nombre spécifique de messages pour synchroniser avec l'agent Python
                    -- dans un contexte d'apprentissage par renforcement, garantissant que toutes les commandes
                    -- sont traitées avant de passer à l'étape suivante.
                    elseif message_from_python:find("wait_for") then
                        local _str_nb_frames = message_from_python:match("wait_for (%d+)")
                        wait_for_messages = tonumber(_str_nb_frames)
                        if wait_for_messages > 0 then
                            flag_wait_for = true
                            -- On ne rajoute plus +1 pour permettre de ne l'appeler qu'une fois 
                            -- ( Python enverra NB_MESSAGES + 1 au premier tour, puis NB_MESSAGES ensuite )
                        end
                        table.insert(responses, "Wait for " .. tostring(wait_for_messages) .. " messages (ACK)")

                    elseif message_from_python:find("frame_per_step") then
                        local _str = message_from_python:match("frame_per_step (%d+)")
                        frame_per_step = tonumber(_str)
                        table.insert(responses,
                                     "Get messages only each " .. tostring(frame_per_step) .. " frame(s)")

                    elseif message_from_python:find("debug ") then
                        -- On récupère le mot suivant "debug "
                        local debug_cmd = message_from_python:match("debug (%a+)")
                        if debug_cmd == "on" then
                            debug = true
                            table.insert(responses, "Debug mode activé (debug=1)")
                        elseif debug_cmd == "off" then
                            debug = false
                            table.insert(responses, "Debug mode désactivé (debug=0)")
                        else
                            table.insert(responses, "ERR: usage 'debug on' ou 'debug off' uniquement")
                        end
                    else
                        table.insert(responses,
                                     "ERR: COMMANDE NON COMPRISE!" .. tostring(message_from_python))
                        print("ERR: COMMANDE NON COMPRISE! (" .. tostring(message_from_python) .. ")")
                    end
                end

                -- On envoie les réponses si on en a
                if #responses > 0 then
                    if wait_for_messages > 0 then
                        messages_in_game = messages_in_game + #responses
                        if debug then
                            local truncated = {}
                            for _, r in ipairs(responses) do
                                table.insert(truncated, string.sub(r, 1, 22))
                            end
                            local resp = table.concat(truncated, ",")
                            io.write("WaitFor#" .. wait_for_messages .. '/#' .. messages_in_game .. '<#' ..
                                     #responses .. '=' .. table.concat(messages_from_python, ",") ..
                                     '->' .. resp .. "\n")
                        end                       
                    elseif debug then
                        print("Réponses envoyées à Python (pour wait_for=0): " .. table.concat(responses, ","))
                    end
                    send_to_python(responses)
                end

            -- On n'a plus de messages à lire
            elseif messages_in_game >= wait_for_messages then
                -- On sort si on a reçu tous les messages attendus
                if flag_wait_for then flag_wait_for = false end
                break
            end
        end

        -- Incrémentation si on est en mode "wait_for"
        if wait_for_messages > 0 then
            if debug then
                print("Nouvelle Attente de " .. wait_for_messages .. " messages")
            end
            frame_in_game = frame_in_game + 1
            nb_messages_total_in_game = nb_messages_total_in_game + messages_in_game
        end
    end
    draw_on_screen()
end

--------------------------------------------------------------------------------
-- Enregistrement des callbacks
--------------------------------------------------------------------------------
emu.register_frame(main)
