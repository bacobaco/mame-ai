-- Bridge qui récupére des
--    "execute <command>(<value>)"
--    "read_memory <address>"
--    "write_memory <address>(<value>)"
local flag = 0
local frame = 0
local latence = 0.0
local debug = false

local machine = manager.machine
local screen = machine.screens[":screen"]
local cpu = machine.devices[":maincpu"]
local mem = cpu.spaces["program"]
local ioport = machine.ioport
local in0 = ioport.ports[":CONTP1"]
local in1 = ioport.ports[":IN1"]
-- Définitions des ports d'entrée MAME (à adapter à votre configuration)
local P1_left = in0.fields["P1 Left"]
local P1_right = in0.fields["P1 Right"]
local P1_Button_1 = in0.fields["P1 Button 1"]
local P1_start = in1.fields["1 Player Start"]
-- Ajoutez d'autres définitions de ports ici si nécessaire

-------------------SOCKET COMM START
local socket = require("socket")
local host = "127.0.0.1"
local port = 12345

local server = assert(socket.bind(host, port))
local ip, port = server:getsockname()
assert(ip, port)
print("Le serveur est en écoute sur le port " .. port)
server:settimeout(5) -- 5 secondes de timeout
local client = nil

-- La fonction accept_client()
local function accept_client()
    print("En attente d'un client...")
    while not client do
        local new_client, err = server:accept()
        if new_client then
            client = new_client
            client:settimeout(0) -- mode non bloquant
            print("Client connecté")
        else
            socket.sleep(0.1) -- Attendre un peu avant de réessayer (évite une utilisation excessive du processeur)
        end
    end
end
accept_client()

local function send_to_python(messages)
    local message_str = table.concat(messages, "\n") .. "\n__end__\n"
    client:send(message_str)
    if debug then
        print(message_str)
    end
end

local function receive_from_python()
    if not client then
        return {}
    end
    local messages = {}
    local data, err = client:receive('*l')
    while data do
        if data == "__end__" then
            break
        end
        table.insert(messages, data)
        data, err = client:receive('*l')
    end
    return messages
end
--------------------SOCKET COMM END

local function execute_command(command, value)
    if command == "P1_left" then
        P1_left:set_value(tonumber(value))
    elseif command == "P1_right" then
        P1_right:set_value(tonumber(value))
    elseif command == "P1_start" then
        P1_start:set_value(tonumber(value))
    elseif command == "P1_Button_1" then
        P1_Button_1:set_value(tonumber(value))
    elseif command == "throttle_rate" then
        machine.video.throttle_rate = tonumber(value)
    elseif command == "throttled" then
        machine.video.throttled = tonumber(value)
        -- ajoutez d'autres commandes ici si n�cessaire
    else
        print(command, value)
        send_to_python({"ERR: execute_command:" .. command .. " non comprise!"})
    end
end

local function read_memory_value(address)
    return mem:read_u8(address)
end
local function write_memory_value(address, value)
    return mem:write_u8(address, value)
end

local nb_messages_total_in_game = 0
local frame_in_game = 0
local wait_for_messages = 0
local frame_per_step = 1
local x, y, _str = 40, 10, ""
-- Le code principal
local function main()
    frame = frame + 1
    if frame % 10000 == 0 then
        print("all frames:" .. frame .. " - " .. "frames_in_game:" .. frame_in_game .. " - " ..
                  tostring(nb_messages_total_in_game) .. " messages traites soit " ..
                  tostring(nb_messages_total_in_game * 100 // frame_in_game / 100) .. " asks/frame_in_game " .. "[" ..
                  tostring(nb_messages_total_in_game * 100 // frame / 100) .. "/frame]")
    end
    if frame % frame_per_step == 0 then
        local messages_in_game = 0
        while true do
            local messages_from_python = receive_from_python()
            local responses = {}
            if #messages_from_python > 0 then
                for _, message_from_python in ipairs(messages_from_python) do
                    if message_from_python:find("execute") then
                        local command, value = message_from_python:match("execute ([%w_]+)%(([%d.]+)%)")
                        execute_command(command, tonumber(value))
                        table.insert(responses, "Command " .. message_from_python .. " ack")
                    elseif message_from_python:find("read_memory") then
                        local address = message_from_python:match("read_memory (%x+)")
                        local memory_value = read_memory_value(tonumber(address, 16))
                        table.insert(responses, tostring(memory_value))
                    elseif message_from_python:find("write_memory") then
                        local address, value = message_from_python:match("write_memory ([%w_]+)%((%d+)%)")
                        local memory_value = write_memory_value(tonumber(address, 16), tonumber(value))
                        table.insert(responses, "Write "..address.."="..value.." ack")
                    elseif message_from_python:find("draw_text") then
                        x, y, _str = message_from_python:match("draw_text%((%d+),(%d+),\"([^\"]+)\"%)")
                        table.insert(responses, "draw_text command ack")
                    elseif message_from_python:find("wait_for") then
                        local _str_nb_frames = message_from_python:match("wait_for (%d+)")
                        wait_for_messages = tonumber(_str_nb_frames)
                        table.insert(responses, "Wait for " .. tostring(wait_for_messages) .. " messages")
                    elseif message_from_python:find("frame_per_step") then
                        local _str = message_from_python:match("frame_per_step (%d+)")
                        frame_per_step = tonumber(_str)
                        table.insert(responses, "Get messages only each " .. tostring(frame_per_step) .. " frame(s)")
                    else
                        table.insert(responses, "ERR: COMMANDE NON COMPRISE!" .. tostring(message_from_python))
                        print("ERR: COMMANDE NON COMPRISE! (" .. tostring(message_from_python) .. ")")
                    end
                end
                if #responses > 0 then
                    if wait_for_messages > 0 then
                        messages_in_game = messages_in_game + #responses
                        if debug then
                            io.write(messages_in_game .. ',')
                        end
                    end
                    send_to_python(responses)
                end
            elseif messages_in_game >= wait_for_messages then
                break
            end
        end
        if wait_for_messages > 0 then
            frame_in_game = frame_in_game + 1
            nb_messages_total_in_game = nb_messages_total_in_game + messages_in_game
        end
    end
end

local function draw_on_screen()
    screen:draw_text(tonumber(x), tonumber(y), _str)
end
-- emu.register_periodic(main)
emu.register_frame_done(draw_on_screen)
emu.register_frame(main)
