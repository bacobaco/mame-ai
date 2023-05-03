-- Paramètres
local alpha = 1 -- 0.1 par défaut. Mise à jour des Q plus fort  0 à 1
local gamma = 1 -- 0.99 par défaut. Maximiser les récompense future 0 à 1
local epsilon = 0.01 -- 0.1 par défaut. Exploitation  => Exploration de 0 => 1
local state_factor = 5 -- Choisissez un facteur approprié pour réduire le nombre d'états
local nb_frame_max_par_state = 5 -- on ne prends en compte qu'une frame sur nb_frame_max_par_state
local actions = {"nope", "right", "left"}

-- Mame Lua Ref
M = {}
local machine = manager.machine
local screen = machine.screens[":screen"]
local cpu = machine.devices[":maincpu"]
local mem = cpu.spaces["program"]
local ioport = machine.ioport
local in0 = ioport.ports[":CONTP1"]
local in1 = ioport.ports[":IN1"]
-- Variables de contrôle
M.P1_left = {
    field = in0.fields["P1 Left"]
}
M.P1_right = {
    field = in0.fields["P1 Right"]
}
M.P1_fire = {
    field = in0.fields["P1 Button 1"]
}
M.start = {
    field = in1.fields["1 Player Start"]
}

local state, next_state
local action_idx
local frame = 0
local partie = 0
local fire_toggle = false
local local_HiScore = 0
local enemy_shots = {}
local last_dir = "right"
local old_score = 0
local mean_score = 0
-- Ajoutez elapsed_time en haut du fichier, après la déclaration des variables
local elapsed_time = 0
local nb_tirs = 0
local remind_nb_ship = 2

-- Variables de jeu invaders
local numAliens = 0x2082
local numCoins = 0x20EB
local P1ScorL = 0x20F8
local P1ScorM = 0x20F9
local p1ShipsRem = 0x21FF -- Ships remaining after current dies
local player1Alive = 0x20E7
local playerAlive = 0x2015 --	Player is alive (FF=alive). Toggles between 0 and 1 for blow-up images.
local playerXr = 0x201B -- Déscripteur de sprite du joueur ... position MSB
local gameMode = 0x20EF
local HiScorL = 0x20F4
local HiScorM = 0x20F5
local shotSync = 0x2080 -- 2080	shotSync	Les 3 tirs sont synchronisés avec le minuteur GO-2. Ceci est copié à partir du minuteur dans la boucle de jeu
local alienShotYr = 0x207B -- 207B	alienShotYr	Delta Y du tir alien
local alienShotXr = 0x207C -- 207C	alienShotXr	Delta X du tir alien

local enemy_shots = {}
local function update_enemy_shots()
    local shot_x = mem:read_u8(alienShotXr)
    local shot_y = mem:read_u8(alienShotYr)
    local sync = mem:read_u8(shotSync)

    enemy_shots[sync + 1] = {
        x = shot_x,
        y = shot_y
    }
    -- print(sync, enemy_shots[sync + 1].x, enemy_shots[sync + 1].y)
end

-- Créez la table Q pour stocker les valeurs Q pour chaque état-action
local Q = {}
local q_state_count = 0 -- Ajoutez cette ligne en haut du fichier, après la déclaration des variables
local function init_q_table(state)
    if Q[state] == nil then
        Q[state] = {}
        for i = 1, #actions do
            Q[state][i] = 0
        end
        q_state_count = q_state_count + 1 -- Incrémentez le compteur lorsque vous ajoutez un nouvel état
        if q_state_count % 100 == 0 then
            hi_score = ((mem:read_u8(HiScorL) >> 4) * 10 + (mem:read_u8(HiScorM) & 0x0F) * 100 +
                           (mem:read_u8(HiScorM) >> 4) * 1000)
            print(os.date("%c") .. ">HiScore pour " .. tostring(q_state_count) .. " states = " .. tostring(hi_score) ..
                      " (Score moyen pour " .. tostring(partie) .. " = " .. tostring(mean_score) .. ")")
            mean_score = 0
            partie = 0
        end
    end
end

-- Fonction pour obtenir l'index d'action avec la Q-value maximale pour un état donné
local function max_action(state)
    init_q_table(state)
    local max_q = -math.huge
    local max_idx = 1
    for i, action in ipairs(actions) do
        local q = Q[state][i] or 0
        if q > max_q then
            max_q = q
            max_idx = i
        end
    end

    return max_idx
end

-- Fonction pour mettre à jour la table Q
local function update_q_table(state, action_idx, reward, next_state)
    init_q_table(state)
    init_q_table(next_state)
    local max_q_next = Q[next_state][max_action(next_state)] or 0
    local q = Q[state][action_idx] or 0

    -- Mise à jour de la valeur Q
    Q[state][action_idx] = q + alpha * (reward + gamma * max_q_next - q)
end

-- Fonction pour choisir une action en utilisant la politique epsilon-greedy
local function choose_action(state)
    -- Exploration
    if math.random() < epsilon then
        return math.random(#actions)
    end
    -- Exploitation
    return max_action(state)
end

local function get_state()
    if mem:read_u8(player1Alive) == 0 then
        mem:write_u8(gameMode, 0)
        return "game_over"
    end
    update_enemy_shots()
    local px = math.floor(mem:read_u8(playerXr) / state_factor)
    -- Trouver la bombe alien la plus proche du joueur en coordonnée x
    local closest_shot = nil
    local min_distance = math.huge
    for i = 1, #enemy_shots do
        local distance = math.abs(enemy_shots[i].x - mem:read_u8(playerXr))
        if distance < min_distance then
            min_distance = distance
            closest_shot = enemy_shots[i]
        end
    end
    if closest_shot ~= nil then
        local shot_state = "(" .. tostring(math.floor(closest_shot.x / state_factor)) .. "," ..
                               tostring(math.floor(closest_shot.y / state_factor)) .. ")"
        return shot_state .. "_(" .. tostring(px) .. ")"
    else
        return "no_bomb_(" .. tostring(px) .. ")"
    end
end

local function send_commands(action)
    if action == "left" then
        M.P1_left.field:set_value(1)
        M.P1_right.field:set_value(0)
    elseif action == "right" then
        M.P1_left.field:set_value(0)
        M.P1_right.field:set_value(1)
    else
        M.P1_left.field:set_value(0)
        M.P1_right.field:set_value(0)
    end
end

local function get_reward()
    score = (mem:read_u8(P1ScorL) >> 4) * 10 + (mem:read_u8(P1ScorM) & 0x0F) * 100 + (mem:read_u8(P1ScorM) >> 4) * 1000
    local time_reward = elapsed_time // 100
    local tirs_reward = nb_tirs // 10
    local ship_dead = (mem:read_u8(p1ShipsRem) - remind_nb_ship) * 500
    local playerVivant = mem:read_u8(playerVivant) 
    local rewards = score - old_score -- - ship_dead -- + time_reward - tirs_reward 
    old_score = score
    remind_nb_ship = mem:read_u8(p1ShipsRem)
    return rewards
end

-- On lance le jeu 
mem:write_u8(numCoins, 0x01)
M.start.field:set_value(1)
-- machine.video.throttle_rate = 0.1
machine.video.throttled = false
-- Fonction principale
local function main()
    if (frame % nb_frame_max_par_state) == 0 and mem:read_u8(gameMode) == 1 then
        -- Incrémentez le temps écoulé à chaque frame:
        elapsed_time = elapsed_time + 1
        -- Récupérer l'état actuel (positions des bombes, soucoupe, joueur)
        state = get_state() -- Cette fonction doit être implémentée pour lire l'état du jeu
        if state == "game_over" then
            -- Réinitialiser le jeu et passer à la prochaine génération
            mean_score = mean_score * partie
            partie = partie + 1
            mean_score = math.floor((mean_score + old_score) / partie)
            old_score = 0
            elapsed_time = 0
            nb_tirs = 0
            mem:write_u8(numCoins, 0x01)
        else
            -- Choisir une action à effectuer
            init_q_table(state)
            action_idx = choose_action(state)

            -- Envoyer les commandes correspondantes au jeu
            send_commands(actions[action_idx])

            -- Récupérer le nouvel état et la récompense (score)
            next_state = get_state() -- Cette fonction doit être implémentée pour lire l'état du jeu
            reward = get_reward() -- Cette fonction doit être implémentée pour lire le score du jeu
            -- io.write(reward,',')
            -- Mettre à jour la table Q
            update_q_table(state, action_idx, reward, next_state)

            -- Préparer la prochaine itération
            state = next_state
            -- print(frame)
        end
    end
    b = io.read()
    screen:draw_text(40, 10,
        "Qsize#: " .. tostring(q_state_count) .. "\nNbPartie#: " .. tostring(partie) .. "\nframe%: " .. tostring(frame) ..
            "\nmean score: " .. tostring(mean_score) .. "\n ships:" .. tostring(mem:read_u8(p1ShipsRem)) ..
            "\n playerAlive:" .. tostring(mem:read_u8(playerAlive)) .. "\nKey?=" ..
            tostring(b))
    fire_toggle = not fire_toggle
    M.P1_fire.field:set_value(fire_toggle and 1 or 0)
    frame = frame + 1
end

local function save_q_table(filename)
    local file = io.open(filename, "w")
    if file then
        for state, action_values in pairs(Q) do
            file:write(state .. "\n")
            for action, value in ipairs(action_values) do
                file:write(tostring(action) .. " " .. tostring(value) .. "\n")
            end
        end
        file:close()
    else
        print("Erreur lors de la sauvegarde de la table Q dans le fichier " .. filename)
    end
end

local function load_q_table(filename)
    local file = io.open(filename, "r")
    if file then
        Q = {}
        q_state_count = 0 -- Réinitialisez q_state_count à 0
        local state = nil
        for line in file:lines() do
            if not string.find(line, " ") then
                state = line
                Q[state] = {}
                q_state_count = q_state_count + 1 -- Incrémentez q_state_count pour chaque état chargé
            else
                local action, value = line:match("([^ ]+) ([^ ]+)")
                action = tonumber(action)
                value = tonumber(value)
                Q[state][action] = value
            end
        end
        file:close()
    else
        print("Erreur lors de la lecture de la table Q à partir du fichier " .. filename)
    end
end

local q_table_filename = "q_table.txt"
load_q_table(q_table_filename)

local function on_exit()
    save_q_table(q_table_filename)
    print("Table Q sauvegardée dans " .. q_table_filename)
end
emu.register_stop(on_exit, "stop")
-- Exécutez la boucle principale à chaque frame
emu.register_frame_done(main) -- , "frame")

-- emu.register_periodic(function() print("callback test") end)
