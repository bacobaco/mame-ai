local lfs = require("lfs")
local debug = false

local M = {}
local machine = manager.machine
local screen = machine.screens[":screen"]
local cpu = machine.devices[":maincpu"]
local mem = cpu.spaces["program"]
local ioport = machine.ioport

local in0 = ioport.ports[":CONTP1"]
local in1 = ioport.ports[":IN1"]

-- Variables de jeu
local numAliens = 0x2082
local numCoins = 0x20EB
local P1ScorL = 0x20F8
local P1ScorM = 0x20F9
local player1Alive = 0x20E7
local playerAlive = 0x2015 --  # Player is alive [FF=alive]. Toggles between 0 and 1 for blow-up images.
local playerXr = 0x201B -- 201B	playerXr	Déscripteur de sprite du joueur ... position MSB
local gameMode = 0x20EF
local HiScorL = 0x20F4
local HiScorM = 0x20F5
-- 2080	shotSync	Les 3 tirs sont synchronisés avec le minuteur GO-2. Ceci est copié à partir du minuteur dans la boucle de jeu
local shotSync = 0x2080
-- 207B	alienShotYr	Delta Y du tir alien
local alienShotYr = 0x207B
-- 207C	alienShotXr	Delta X du tir alien
local alienShotXr = 0x207C
local rolShotYr = 0x203D -- # Game object 2: Alien rolling-shot (targets player specifically)
local rolShotXr = 0x203E
local squShotYr = 0x205D -- # Game object 4: squiggly shot position
local squShotXr = 0x205E
local pluShotYr = 0x204D -- # the plunger-shot (object 3) position
local pluSHotXr = 0x204E

-- Variables de contrôle
M.P1_left = {
    in0 = in0,
    field = in0.fields["P1 Left"]
}
M.P1_right = {
    in0 = in0,
    field = in0.fields["P1 Right"]
}
M.P1_fire = {
    in0 = in0,
    field = in0.fields["P1 Button 1"]
}
M.start = {
    in1 = in1,
    field = in1.fields["1 Player Start"]
}

-- Paramètres de l'algorithme génétique
local taille_population = 100
local taux_de_mutation = 0.1

-- Autres variables globales
local nb_aliens_restants
local nb_aliens_restants_old = 56
local population = {}
local individu_actuel = 1
local frame = 1
local nb_frames_par_gene = 1
local generation = 1
local fire_toggle = false
local local_HiScore = 0
local last_action = 0
local last_dir = "left"
local enemy_shots = {}
local mean_score = 0

local function update_enemy_shots()
    local rx = mem:read_u8(rolShotXr)
    local ry = mem:read_u8(rolShotYr)
    local sx = mem:read_u8(squShotXr)
    local sy = mem:read_u8(squShotYr)
    local px = mem:read_u8(pluSHotXr)
    local py = mem:read_u8(pluShotYr)

    enemy_shots = {{
        x = rx,
        y = ry
    }, {
        x = sx,
        y = sy
    }, {
        x = px,
        y = py
    }}
    -- print(sync, enemy_shots[sync + 1].x, enemy_shots[sync + 1].y)
end
local function test_bombes()
    local px = mem:read_u8(playerXr)
    -- Pour lire la valeur du port entier
    local valeurPort = in0:read()

    -- Pour extraire l'état des bits correspondants aux champs "P1 Left" et "P1 Right"
    local valeurP1Left = (valeurPort & M.P1_left.field.mask) ~= 0 and true or false
    local valeurP1Right = (valeurPort & M.P1_right.field.mask) ~= 0 and true or false

    for i = 1, #enemy_shots do
        -- print(tostring(enemy_shots[i].x), tostring(px - 1), tostring(px + 13), tostring(enemy_shots[i].x >= px - 1),
        --     tostring(enemy_shots[i].x <= px + 13), tostring(enemy_shots[i].y >= 0), tostring(enemy_shots[i].y <= 100),
        --     tostring(valeurP1Left), tostring(valeurP1Right),tostring(true and not valeurP1Left and not valeurP1Right))

        if (enemy_shots[i].x >= px - 1 and enemy_shots[i].x <= px + 13) and
            (enemy_shots[i].y >= 0 and enemy_shots[i].y <= 100) then
            return (true and not valeurP1Left and not valeurP1Right) -- si bombes proche et même colonne
        end
    end
    return (false)
end
-- Fonction pour sélectionner un individu en fonction du score (plus grand score a une meilleure chance d'être sélectionné)
local function selection_ponderee(population)
    local total_score = 0
    for _, individu in ipairs(population) do
        total_score = total_score + individu.score
    end
    local rnd = math.random() * total_score
    local somme_score = 0
    for _, individu in ipairs(population) do
        somme_score = somme_score + individu.score
        if rnd <= somme_score then
            return individu
        end
    end
end
-- Trie la population en fonction du score et renvoie les n meilleurs individus
local function selection_triee(population, n)
    -- Tri la population par score décroissant
    table.sort(population, function(a, b)
        return a.score > b.score
    end)
    -- Sélectionne les n meilleurs individus
    local meilleurs_individus = {}
    for i = 1, n do
        table.insert(meilleurs_individus, population[i])
    end

    return meilleurs_individus
end

-- Croisement
local function croisement(parent1, parent2)
    local enfant = {
        genes = {}
    }
    -- on prends 10x10frames du gène du parent1 puis on prends le parent2 etc...
    local segment_longueur = 10

    local parent_selectionne = parent1
    local j = 1

    for i = 1, #parent1.genes do
        enfant.genes[i] = parent_selectionne.genes[i]

        j = j + 1
        if j > segment_longueur then
            j = 1
            parent_selectionne = parent_selectionne == parent1 and parent2 or parent1
        end
    end

    return enfant
end

-- Mutation
local function mutation(individu, taux)
    for i = 1, #individu.genes do
        if math.random() < taux then
            individu.genes[i] = math.random(0, 1)
        end
    end
end
-- Créer la population initiale
for i = 1, taille_population do
    local individu = {
        genes = {}
    }
    -- un tableau dure autour de 10000 frames
    for j = 1, 10000 / nb_frames_par_gene do
        individu.genes[j] = math.random(0, 1)
    end
    table.insert(population, individu)
end

local function reset_invaders()
    --    machine:soft_reset()
    mem:write_u8(numCoins, 0x01)
    M.start.field:set_value(1)
    machine.video.throttled = true
    machine.video.throttle_rate = 5
end
function sauvegarderTable(tableau, nomFichier)
    local fichier = io.open(nomFichier, "w")

    fichier:write(generation .. "\n")
    for i = 1, #tableau do
        for j = 1, #tableau[i].genes do
            fichier:write(tableau[i].genes[j])
            if j ~= #tableau[i].genes then
                fichier:write(",")
            end
        end
        fichier:write("\n")
    end

    fichier:close()
end
function chargerTable(nomFichier)
    local tableau = {}
    local fichier = io.open(nomFichier, "r")
    local i = 1

    generation = tonumber(fichier:read("*l"))
    for ligne in fichier:lines() do
        tableau[i] = {
            genes = {}
        }
        local j = 1
        for valeur in string.gmatch(ligne, "([^,]+)") do
            tableau[i].genes[j] = tonumber(valeur)
            j = j + 1
        end
        i = i + 1
    end

    fichier:close()
    return tableau
end
function fichierExiste(nomFichier)
    local fichier = io.open(nomFichier, "r")
    if fichier ~= nil then
        io.close(fichier)
        return true
    else
        return false
    end
end
if fichierExiste("population.txt") then
    chargerTable("population.txt")
end
reset_invaders()
function M.run_ai()
    -- prendre les coord. des bombes aliens
    update_enemy_shots()
    -- Obtenir le nombre d'aliens restants
    nb_aliens_restants = mem:read_u8(numAliens)

    if mem:read_u8(gameMode) == 1 then
        -- Utiliser les gènes de l'individu actuel pour choisir l'action s'il existe sinon créer
        local action = population[individu_actuel].genes[(frame // nb_frames_par_gene) % (10000 / nb_frames_par_gene)]
        -- print("action="..tostring(action).." for frame//10="..tostring((frame // 10) % 1000))
        if last_action ~= action then -- action=0 => changer de direction
            if action == 0 or test_bombes() or nb_aliens_restants_old == nb_aliens_restants then
                if last_dir == "right" then
                    M.P1_left.field:set_value(1)
                    M.P1_right.field:set_value(0)
                    last_dir = "left"
                else
                    M.P1_left.field:set_value(0)
                    M.P1_right.field:set_value(1)
                    last_dir = "right"
                end
            else
                M.P1_left.field:set_value(0)
                M.P1_right.field:set_value(0)
            end
            last_action = action
        end
        fire_toggle = not fire_toggle
        M.P1_fire.field:set_value(fire_toggle and 1 or 0)
        -- emu.step()
        frame = frame + 1
        nb_aliens_restants_old = nb_aliens_restants
        -- Si le joueur n'a plus de vies, passer à l'individu suivant
        if mem:read_u8(player1Alive) == 0 then
            mem:write_u8(numCoins, 0x01)
            mem:write_u8(gameMode, 0)
            -- Évaluer l'individu actuel
            local score = (mem:read_u8(P1ScorL) >> 4) * 10 + (mem:read_u8(P1ScorM) & 0x0F) * 100 +
                              (mem:read_u8(P1ScorM) >> 4) * 1000
            local_HiScore = math.max(score, local_HiScore)

            population[individu_actuel].score = score --  - nb_aliens_restants * 5 < 0 and 0 or score - nb_aliens_restants * 5
            individu_actuel = individu_actuel + 1
            mean_score = ((individu_actuel - 1) * mean_score + score) // individu_actuel -- à chaque nouvelle génération on reprendra la moyenne 
            frame = 1
            -- Si tous les individus ont été évalués, passer à la prochaine génération
            if individu_actuel > taille_population then
                print("GENERATION#" .. tostring(generation) .. ": Mean-Score=" .. mean_score .. " - Local-Hiscore=" ..
                          local_HiScore)
                local_HiScore = 0
                local nouvelle_population = {}
                for i = 1, taille_population do
                    meilleurs = selection_triee(population, 1)
                    -- local parent1 = meilleurs[1]
                    local parent2 = meilleurs[1]
                    local parent1 = selection_ponderee(population)
                    -- local parent2 = selection_ponderee(population)
                    local enfant = croisement(parent1, parent2)
                    mutation(enfant, taux_de_mutation)
                    table.insert(nouvelle_population, enfant)
                end
                population = nouvelle_population
                individu_actuel = 1
                generation = generation + 1
                reset_invaders()
                sauvegarderTable(population, "population.txt")
            end
        end
    end

    if debug then
        str_bombes = ""
        for i = 1, #enemy_shots do
            str_bombes = str_bombes .. tostring(i) .. "=(" .. tostring(enemy_shots[i].x) .. "," ..
                             tostring(enemy_shots[i].y .. "),")
        end
        screen:draw_text(40, 10,
            "Generation: " .. tostring(generation) .. "\nIndividu: " .. tostring(individu_actuel) .. "\nPlayerAlive?: " ..
                tostring(mem:read_u8(player1Alive)) .. "\nlocal_HiScore: " .. tostring(local_HiScore) .. "\nFrames: " ..
                tostring(frame) .. "\nbombes enemy:" .. tostring(str_bombes) .. "\nTestBombes?:" ..
                tostring(test_bombes()) .. "\nPlayerX:" .. tostring(mem:read_u8(playerXr)))
    else
        screen:draw_text(40, 1, "Generation: " .. tostring(generation) .. "\nIndividu: " .. tostring(individu_actuel) ..
            "\nMeanScore pour cette Génération: " .. tostring(mean_score))
    end
end

emu.register_frame_done(M.run_ai, "frame")
