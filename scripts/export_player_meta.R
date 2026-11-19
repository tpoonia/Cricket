# Export cricketdata's bundled player_meta (lazy data, NOT scraped) to CSV.
# Requires: install.packages("cricketdata"). Run from repo root:
#   Rscript scripts/export_player_meta.R
suppressMessages(library(cricketdata))
data("player_meta", package = "cricketdata")
pm <- as.data.frame(player_meta)
dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)
write.csv(pm, "data/processed/player_meta.csv", row.names = FALSE, na = "")
cat("wrote", nrow(pm), "rows to data/processed/player_meta.csv\n")
