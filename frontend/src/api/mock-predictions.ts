// Mock data for the prototype. In the real app these come from the
// FastAPI/whatever endpoint that serves the YOLOv11 model.
import type { Prediction, SampleFish } from "./types";

export const SAMPLE_FISH: SampleFish[] = [
  { id: "clownfish",    label: "Clownfish",     hue: 28,  caption: "ANEMONEFISH" },
  { id: "lionfish",     label: "Lionfish",      hue: 12,  caption: "LIONFISH"    },
  { id: "moorish-idol", label: "Moorish Idol",  hue: 55,  caption: "MOORISH IDOL" },
  { id: "manta",        label: "Manta Ray",     hue: 220, caption: "MANTA RAY"   },
  { id: "yellow-tang",  label: "Yellow Tang",   hue: 48,  caption: "YELLOW TANG" }
];

// Each prediction has top-3 across species / genus / family + a habitat blurb +
// a (mock) bounding box expressed as % of the rendered image.
export const MOCK_PREDICTIONS: Record<string, Prediction> = {
  "clownfish": {
    summary: "Clown anemonefish",
    common: "Ocellaris clownfish",
    bbox: { x: 28, y: 32, w: 44, h: 42 },
    habitat: "Lives in symbiosis with sea anemones across warm, shallow lagoons of the Indo-Pacific. Most often spotted at 1–15 m on coral reefs in eastern Indonesia, Papua New Guinea and the northern Australian coast.",
    species: [
      { name: "Amphiprion ocellaris",  conf: 94.2 },
      { name: "Amphiprion percula",    conf:  3.8 },
      { name: "Premnas biaculeatus",   conf:  1.2 }
    ],
    genus: [
      { name: "Amphiprion",  conf: 98.0 },
      { name: "Premnas",     conf:  1.5 },
      { name: "Dascyllus",   conf:  0.4 }
    ],
    family: [
      { name: "Pomacentridae",   conf: 99.1 },
      { name: "Labridae",        conf:  0.6 },
      { name: "Chaetodontidae",  conf:  0.2 }
    ]
  },
  "lionfish": {
    summary: "Red lionfish",
    common: "Red lionfish",
    bbox: { x: 18, y: 22, w: 60, h: 60 },
    habitat: "Indo-Pacific native, now invasive across the Caribbean and Western Atlantic. Hunts at dusk along reef ledges and wrecks between 2 and 50 m.",
    species: [
      { name: "Pterois volitans",   conf: 88.7 },
      { name: "Pterois miles",      conf:  9.1 },
      { name: "Pterois antennata",  conf:  1.6 }
    ],
    genus: [
      { name: "Pterois",       conf: 99.4 },
      { name: "Dendrochirus",  conf:  0.4 },
      { name: "Scorpaena",     conf:  0.1 }
    ],
    family: [
      { name: "Scorpaenidae",  conf: 99.7 },
      { name: "Synanceiidae",  conf:  0.2 },
      { name: "Tetrarogidae",  conf:  0.1 }
    ]
  },
  "moorish-idol": {
    summary: "Moorish idol",
    common: "Moorish idol",
    bbox: { x: 22, y: 18, w: 50, h: 64 },
    habitat: "The only living member of its family. Cruises clear tropical reefs from East Africa to Hawaii in pairs or small schools, usually between 3 and 25 m.",
    species: [
      { name: "Zanclus cornutus",       conf: 96.3 },
      { name: "Heniochus diphreutes",   conf:  2.4 },
      { name: "Heniochus acuminatus",   conf:  1.1 }
    ],
    genus: [
      { name: "Zanclus",     conf: 96.8 },
      { name: "Heniochus",   conf:  2.9 },
      { name: "Forcipiger",  conf:  0.2 }
    ],
    family: [
      { name: "Zanclidae",        conf: 92.1 },
      { name: "Chaetodontidae",   conf:  7.5 },
      { name: "Acanthuridae",     conf:  0.3 }
    ]
  },
  "manta": {
    summary: "Reef manta ray",
    common: "Reef manta ray",
    bbox: { x: 8,  y: 28, w: 80, h: 50 },
    habitat: "Filter-feeds on plankton at cleaning stations and current-swept reef edges. Common across Indo-Pacific dive sites such as Komodo, Maldives, and the Yasawas.",
    species: [
      { name: "Mobula alfredi",         conf: 76.4 },
      { name: "Mobula birostris",       conf: 21.8 },
      { name: "Mobula tarapacana",      conf:  1.3 }
    ],
    genus: [
      { name: "Mobula",     conf: 99.6 },
      { name: "Rhinoptera", conf:  0.3 },
      { name: "Aetobatus",  conf:  0.1 }
    ],
    family: [
      { name: "Mobulidae",   conf: 99.6 },
      { name: "Myliobatidae",conf:  0.3 },
      { name: "Rhinopteridae", conf: 0.1 }
    ]
  },
  "yellow-tang": {
    summary: "Yellow tang",
    common: "Yellow tang",
    bbox: { x: 24, y: 30, w: 52, h: 44 },
    habitat: "Endemic to the central Pacific, especially the lava reefs around the Hawaiian Islands. Grazes algae in 2–46 m, often in loose aggregations.",
    species: [
      { name: "Zebrasoma flavescens",   conf: 91.5 },
      { name: "Zebrasoma scopas",       conf:  6.7 },
      { name: "Zebrasoma xanthurum",    conf:  1.2 }
    ],
    genus: [
      { name: "Zebrasoma",   conf: 98.2 },
      { name: "Acanthurus",  conf:  1.5 },
      { name: "Ctenochaetus",conf:  0.3 }
    ],
    family: [
      { name: "Acanthuridae",   conf: 99.4 },
      { name: "Pomacanthidae",  conf:  0.4 },
      { name: "Siganidae",      conf:  0.2 }
    ]
  }
};

export const DEFAULT_PREDICTION_KEY = "clownfish";
