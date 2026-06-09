/* Field-log mock data — collected species + per-sighting records.
   Real app reads this from local DB / API. */

export const LOG_TOTAL_SPECIES = 1247;

export const LOG_SPECIES = [
  {
    no: 1, id: "amphiprion-ocellaris",
    common: "Ocellaris clownfish",
    species: "Amphiprion ocellaris",
    genus: "Amphiprion",
    family: "Pomacentridae",
    hue: 28,
    caption: "ANEMONEFISH",
    bestConf: 96.4,
    habitat: "Symbiotic with sea anemones across warm Indo-Pacific lagoons. Most often spotted at 1–15 m on coral reefs in eastern Indonesia, PNG and northern Australia.",
    sightings: [
      { date: "2026-04-22", site: "Tulamben — House Reef",  region: "Bali, Indonesia",        depth: 8,  tempC: 28, conf: 96.4 },
      { date: "2026-04-19", site: "Mimpang",                 region: "Bali, Indonesia",        depth: 14, tempC: 27, conf: 92.8 },
      { date: "2026-02-11", site: "Apo Reef",                region: "Mindoro, Philippines",   depth: 6,  tempC: 29, conf: 89.1 },
      { date: "2026-02-09", site: "Sabang Wrecks",           region: "Mindoro, Philippines",   depth: 12, tempC: 29, conf: 94.3 },
      { date: "2025-11-04", site: "Pulau Sipadan",           region: "Sabah, Malaysia",        depth: 5,  tempC: 28, conf: 95.7 },
      { date: "2025-11-02", site: "Mabul — Eel Garden",      region: "Sabah, Malaysia",        depth: 9,  tempC: 28, conf: 88.6 },
      { date: "2025-08-21", site: "Lady Elliot Island",      region: "QLD, Australia",         depth: 11, tempC: 24, conf: 90.4 },
      { date: "2025-06-30", site: "Heron Bommie",            region: "QLD, Australia",         depth: 7,  tempC: 23, conf: 86.9 }
    ]
  },
  {
    no: 2, id: "pterois-volitans",
    common: "Red lionfish",
    species: "Pterois volitans",
    genus: "Pterois",
    family: "Scorpaenidae",
    hue: 12,
    caption: "LIONFISH",
    bestConf: 94.0,
    habitat: "Indo-Pacific native; invasive across the Caribbean. Cruises ledges and wrecks at dusk between 2 and 50 m.",
    sightings: [
      { date: "2026-04-20", site: "USAT Liberty Wreck",   region: "Bali, Indonesia",     depth: 22, tempC: 26, conf: 94.0 },
      { date: "2026-02-10", site: "Sabang Wrecks",         region: "Mindoro, Philippines",depth: 18, tempC: 28, conf: 91.6 },
      { date: "2025-08-19", site: "Yongala Wreck",         region: "QLD, Australia",      depth: 27, tempC: 23, conf: 87.2 }
    ]
  },
  {
    no: 3, id: "mobula-alfredi",
    common: "Reef manta ray",
    species: "Mobula alfredi",
    genus: "Mobula",
    family: "Mobulidae",
    hue: 220,
    caption: "MANTA RAY",
    bestConf: 88.5,
    habitat: "Plankton filter-feeder. Found at cleaning stations and current-swept reef edges across the Indo-Pacific.",
    sightings: [
      { date: "2026-04-21", site: "Manta Point",           region: "Nusa Penida, Indonesia", depth: 12, tempC: 24, conf: 88.5 },
      { date: "2025-08-22", site: "Lady Elliot — Lighthouse", region: "QLD, Australia",      depth: 14, tempC: 23, conf: 81.7 }
    ]
  },
  {
    no: 4, id: "zebrasoma-flavescens",
    common: "Yellow tang",
    species: "Zebrasoma flavescens",
    genus: "Zebrasoma",
    family: "Acanthuridae",
    hue: 48,
    caption: "YELLOW TANG",
    bestConf: 95.1,
    habitat: "Endemic to the central Pacific, especially Hawaiian lava reefs. Grazes algae from 2 to 46 m.",
    sightings: [
      { date: "2025-12-12", site: "Two Step",              region: "Big Island, Hawaii",  depth: 7,  tempC: 25, conf: 95.1 },
      { date: "2025-12-11", site: "Kealakekua Bay",        region: "Big Island, Hawaii",  depth: 11, tempC: 25, conf: 92.0 },
      { date: "2025-12-09", site: "Sheraton Caverns",      region: "Kauai, Hawaii",       depth: 18, tempC: 24, conf: 88.6 },
      { date: "2025-12-08", site: "Tunnels Reef",          region: "Kauai, Hawaii",       depth: 9,  tempC: 24, conf: 90.4 },
      { date: "2025-12-07", site: "Kahala Barge",          region: "Oahu, Hawaii",        depth: 24, tempC: 24, conf: 86.2 }
    ]
  },
  {
    no: 5, id: "zanclus-cornutus",
    common: "Moorish idol",
    species: "Zanclus cornutus",
    genus: "Zanclus",
    family: "Zanclidae",
    hue: 55,
    caption: "MOORISH IDOL",
    bestConf: 96.3,
    habitat: "Only living member of its family. Cruises clear tropical reefs East Africa to Hawaii in pairs or small schools.",
    sightings: [
      { date: "2026-04-19", site: "Crystal Bay",           region: "Nusa Penida, Indonesia", depth: 16, tempC: 22, conf: 96.3 },
      { date: "2026-02-09", site: "Verde Island Pinnacle", region: "Batangas, Philippines",  depth: 20, tempC: 27, conf: 90.7 },
      { date: "2025-11-03", site: "Barracuda Point",       region: "Sabah, Malaysia",        depth: 14, tempC: 28, conf: 89.4 },
      { date: "2025-08-20", site: "Stanley Reef",          region: "QLD, Australia",         depth: 10, tempC: 23, conf: 84.5 }
    ]
  },
  {
    no: 6, id: "triaenodon-obesus",
    common: "Whitetip reef shark",
    species: "Triaenodon obesus",
    genus: "Triaenodon",
    family: "Carcharhinidae",
    hue: 200,
    caption: "WHITETIP",
    bestConf: 79.6,
    habitat: "Rests in caves by day, hunts crustaceans and reef fish at night. Indo-Pacific reefs, 8–40 m.",
    sightings: [
      { date: "2025-11-03", site: "Barracuda Point",       region: "Sabah, Malaysia",        depth: 22, tempC: 28, conf: 79.6 }
    ]
  },
  {
    no: 7, id: "chelonia-mydas",
    common: "Green sea turtle",
    species: "Chelonia mydas",
    genus: "Chelonia",
    family: "Cheloniidae",
    hue: 130,
    caption: "GREEN TURTLE",
    bestConf: 98.2,
    habitat: "Herbivorous reptile grazing seagrass beds and algae lawns. Common cleaning-station visitor across tropical seas.",
    sightings: [
      { date: "2026-04-22", site: "Turtle Bay",            region: "Bali, Indonesia",        depth: 6,  tempC: 28, conf: 98.2 },
      { date: "2025-12-10", site: "Tunnels Reef",          region: "Kauai, Hawaii",          depth: 12, tempC: 24, conf: 95.4 },
      { date: "2025-12-09", site: "Sheraton Caverns",      region: "Kauai, Hawaii",          depth: 9,  tempC: 24, conf: 96.8 },
      { date: "2025-11-04", site: "Pulau Sipadan",         region: "Sabah, Malaysia",        depth: 4,  tempC: 28, conf: 92.1 },
      { date: "2025-08-21", site: "Lady Elliot Island",    region: "QLD, Australia",         depth: 8,  tempC: 24, conf: 94.6 },
      { date: "2025-06-29", site: "Heron Bommie",          region: "QLD, Australia",         depth: 6,  tempC: 23, conf: 90.3 }
    ]
  },
  {
    no: 8, id: "octopus-cyanea",
    common: "Day octopus",
    species: "Octopus cyanea",
    genus: "Octopus",
    family: "Octopodidae",
    hue: 290,
    caption: "DAY OCTOPUS",
    bestConf: 71.4,
    habitat: "Diurnal hunter on rubble and reef flats. Notable colour-change camouflage; Indo-Pacific to 50 m.",
    sightings: [
      { date: "2025-12-08", site: "Tunnels Reef",          region: "Kauai, Hawaii",          depth: 11, tempC: 24, conf: 71.4 },
      { date: "2025-11-02", site: "Mabul — Eel Garden",    region: "Sabah, Malaysia",        depth: 10, tempC: 28, conf: 68.9 }
    ]
  },
  {
    no: 9, id: "synchiropus-splendidus",
    common: "Mandarinfish",
    species: "Synchiropus splendidus",
    genus: "Synchiropus",
    family: "Callionymidae",
    hue: 320,
    caption: "MANDARINFISH",
    bestConf: 82.0,
    habitat: "Tiny, jewel-coloured dragonet. Emerges at dusk from staghorn rubble; western Pacific reefs 1–18 m.",
    sightings: [
      { date: "2026-02-09", site: "Sabang — Mandarin Dive", region: "Mindoro, Philippines",  depth: 6,  tempC: 29, conf: 82.0 }
    ]
  },
  {
    no: 10, id: "antennarius-commerson",
    common: "Giant frogfish",
    species: "Antennarius commerson",
    genus: "Antennarius",
    family: "Antennariidae",
    hue: 18,
    caption: "FROGFISH",
    bestConf: 64.8,
    habitat: "Master of disguise; ambush predator perched on sponges and reef walls. Indo-Pacific, 1–75 m.",
    sightings: [
      { date: "2025-11-01", site: "Mabul — Paradise II",   region: "Sabah, Malaysia",        depth: 16, tempC: 28, conf: 64.8 }
    ]
  },
  {
    no: 11, id: "sphaeramia-nematoptera",
    common: "Pyjama cardinalfish",
    species: "Sphaeramia nematoptera",
    genus: "Sphaeramia",
    family: "Apogonidae",
    hue: 75,
    caption: "PYJAMA CARDINAL",
    bestConf: 85.7,
    habitat: "Sheltering in branching corals by day, often in tight schools. Western Pacific, 1–14 m.",
    sightings: [
      { date: "2026-02-10", site: "Sabang Wrecks",         region: "Mindoro, Philippines",   depth: 13, tempC: 29, conf: 85.7 },
      { date: "2025-11-02", site: "Mabul — Eel Garden",    region: "Sabah, Malaysia",        depth: 8,  tempC: 28, conf: 80.1 },
      { date: "2025-11-01", site: "Mabul — Paradise II",   region: "Sabah, Malaysia",        depth: 9,  tempC: 28, conf: 78.4 }
    ]
  },
  {
    no: 12, id: "laticauda-colubrina",
    common: "Banded sea krait",
    species: "Laticauda colubrina",
    genus: "Laticauda",
    family: "Elapidae",
    hue: 240,
    caption: "SEA KRAIT",
    bestConf: 76.2,
    habitat: "Amphibious sea snake hunting eels in reef cracks; surfaces to breathe and rest on shore.",
    sightings: [
      { date: "2025-11-03", site: "Barracuda Point",       region: "Sabah, Malaysia",        depth: 19, tempC: 28, conf: 76.2 }
    ]
  }
];

// Ghost slots representing the next undiscovered species in the diver's region.
export const LOG_GHOSTS = [
  { no: 13, hint: "Pomacanthidae · ?" },
  { no: 14, hint: "Chaetodontidae · ?" },
  { no: 15, hint: "Labridae · ?" },
  { no: 16, hint: "Serranidae · ?" }
];
