# Dataset Notes

## Classification Train/Val Split

**Geographic cluster split** — 80 clusters (k=80, GPU KMeans on lat/lon radians); 20% of clusters
randomly assigned to val, 80% to train. Seed 42.

**Val floor** — any taxon with fewer than 20 natural val examples is topped up by randomly
moving photos from its train pool. Taxa with zero train examples (all photos in val clusters)
are dropped rather than rescued by moving val → train, as doing so would collapse the geographic
separation for those taxa.

**Train floor — 50 examples (114 taxa dropped)**
Taxa with fewer than 50 natural train examples (photos landing in train clusters before any rescue)
are excluded from the dataset entirely. Rescuing them requires moving photos from val clusters into
train, which breaks the geographic split for precisely the most geographically concentrated species.
At <50 examples a pretrained backbone produces unreliable class boundaries even for visually
distinctive species. 114 taxa dropped; ~1,440 retained.

**Dropped taxa are geographically concentrated endemics** — species whose entire iNaturalist
observation footprint falls within a small number of geographic clusters. They are not learnable
under a geographic generalisation constraint without data contamination.
