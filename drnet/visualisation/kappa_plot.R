# Copyright (C) 2019  Patrick Schwab, ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions
#  of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

library(latex2exp)
makeTransparent = function(..., alpha=0.15) {
  # From: https://stackoverflow.com/a/20796068
  if(alpha<0 | alpha>1) stop("alpha must be between 0 and 1")

  alpha = floor(255*alpha)
  newColor = col2rgb(col=unlist(list(...)), alpha=FALSE)
  .makeTransparent = function(col, alpha) {
    rgb(red=col[1], green=col[2], blue=col[3], alpha=alpha, maxColorValue=255)
  }
  newColor = apply(newColor, 2, .makeTransparent, alpha=alpha)
  return(newColor)
}

# The following numbers were extracted from the finished experiments using run_results.sh.
no_tarnet_mse_mise <- c(13.8, 14.3, 15.3, 15.6, 16.0, 16.1, 16.3)
no_tarnet_mse_dpe <- c(43.9, 45.5, 47.1, 47.7, 48.4, 48.7, 49.0)

gps_mse_mise <- c(31.6, 46.2, 47.7, 56.4, 69.6, 93.0, 60.7)
gps_mse_dpe <- c(35.8, 38.9, 42.3, 43.7, 45.2, 45.9, 46.2)

tarnet_mse_mise <- c(7.7, 7.4, 8.1, 8.7, 8.7, 9.1, 8.6)
tarnet_mse_dpe <- c(12.4, 12.8, 13.9, 14.6, 14.6, 15.0, 15.1)

tarnet_no_strata_mse_mise <- c(13.8, 14.6, 15.6, 15.7, 16.3, 16.3, 16.5)
tarnet_no_strata_mse_dpe <- c(39.5, 45.4, 45.1, 48.0, 48.3, 48.6, 45.9)

kappas <- c(5, 7, 10, 12, 15, 17, 20)
mise_min <- 5
mise_max <- 100
mise_title <- TeX("$\\sqrt{MISE}$")
mise_title_cex <- 3

dpe_min <- 10
dpe_max <- 50
dpe_title <- TeX("$\\sqrt{DPE}$")
dpe_title_cex <- 3

metric_idx <- 1
for(metric in c("mise", "dpe")) {
  # Plot setup.
  pdf(paste("kappa_", metric, ".pdf", sep=""), width=14)
  par(mar = c(5.1, 5.6, 4.1, 2.1))

  # Get per-metric config.
  min_y <- get(paste(metric, "_min", sep=""))
  max_y <- get(paste(metric, "_max", sep=""))
  title <- get(paste(metric, "_title", sep=""))
  title_cex <- get(paste(metric, "_title_cex", sep=""))

  plot(kappas, no_tarnet_mse_mise, type = 'n', xlim=c(min(kappas), max(kappas)), ylim=c(min_y, max_y),
       cex.axis=2, ann=FALSE, mgp=c(3, 1, 0), xaxs = "i", yaxs = "i")
  mtext(side=1, text=substitute(paste("Treatment Assignment Bias ", kappa)), line=4, cex=3)
  mtext(side=2, text=title, line=2.75, cex=title_cex)

  # Plot each method.
  i <- 1
  postfix_mean <- paste("_", metric, sep="")
  colors <- c('#F59799', '#9DC7EA', '#FDC67C', '#A75CC6', '#A7916D', '#666666')
  methods <- c("tarnet_mse", "tarnet_no_strata_mse", "no_tarnet_mse", "gps_mse")
  names <- c("DRNet", "TARNET", "MLP", "GPS")
  for(method in methods) {
    mean <- get(paste(method, postfix_mean, sep=""))

    lines(kappas, mean, lwd=10, cex=3, pch=i-1, type="o", col=colors[i])

    i <- i + 1
  }
  if (metric_idx == 1) {
    legend("topleft", legend=names, pch=0:length(methods), col=colors, cex=2, lwd=5)
  }
  dev.off()

  metric_idx <- metric_idx + 1
}
