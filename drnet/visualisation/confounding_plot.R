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

# {9.6} $\pm$ 0.0 &  {2.1} $\pm$ 0.4 & {2.1} $\pm$ 0.1 (0%)
# {11.9} $\pm$ 0.2 & {4.8} $\pm$ 1.8 & {4.8} $\pm$ 2.8 (50%)
# {12.2} $\pm$ 0.2 & {6.2} $\pm$ 2.6 & {5.6} $\pm$ 6.3 (90%)

# The following numbers were extracted from the finished experiments using run_results.sh.
mise <- c(9.6, 11.9, 12.2)
dpe <- c(2.1, 4.8, 6.2)
pe <- c(2.1, 4.8, 5.6)

data <- t(cbind(mise, dpe, pe))
# data <- t(apply(data, 1, function(x)(x/(max(x)))))
rownames(data) <- c("MISE", "DPE", "PE")
colnames(data) <- c("0%", "50%", "80%")

colors <- c('#F59799', '#9DC7EA', '#FDC67C') #, '#666666')

pdf("confounding.pdf", width=10, height=5)
plt = barplot(data, 
        col=colors , 
        density=c(50, 50, 50, 50),
        angle=c(45, 45, 45, -45),
        ylim=c(0, 16),
        beside=T, xlab="Percentage of Hidden Confounding [%]", 
        cex.axis=1.5, cex=2, cex.lab=2)
legend("topright", legend=rownames(data), cex=1.5,
       fill=colors, bty="o")

dev.off()