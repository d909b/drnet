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

# Computation time was extracted from the finished run outputs as follows:
# cat /path/to/drnet_icu3a10k2e_tarnet_mse_1/run_0/*.txt| grep time= |  awk '{print $9}' | grep time= | sed -e 's/time=\(.*\),.*score=.*/\1/' | awk -F : '{sum+=$1} END {print "AVG=",sum/NR}'
# cat /path/to/drnet_icu3a10k4e_tarnet_mse_1/run_0/*.txt| grep time= |  awk '{print $9}' | grep time= | sed -e 's/time=\(.*\),.*score=.*/\1/' | awk -F : '{sum+=$1} END {print "AVG=",sum/NR}'
# cat /path/to/drnet_icu3a10k6e_tarnet_mse_1/run_0/*.txt| grep time= |  awk '{print $9}' | grep time= | sed -e 's/time=\(.*\),.*score=.*/\1/' | awk -F : '{sum+=$1} END {print "AVG=",sum/NR}'
# cat /path/to/drnet_icu3a10k8e_tarnet_mse_1/run_0/*.txt| grep time= |  awk '{print $9}' | grep time= | sed -e 's/time=\(.*\),.*score=.*/\1/' | awk -F : '{sum+=$1} END {print "AVG=",sum/NR}'
# cat /path/to/drnet_icu3a10k10e_tarnet_mse_1/run_0/*.txt| grep time= |  awk '{print $9}' | grep time= | sed -e 's/time=\(.*\),.*score=.*/\1/' | awk -F : '{sum+=$1} END {print "AVG=",sum/NR}'

# The following numbers were extracted from the finished experiments using run_results.sh.
mise <- c(49.4, 44.5, 33.5, 39.0, 35.1)
dpe <- c(106.7, 66.7, 45.6, 48.1, 35.1)
pe <- c(136.9, 102.0, 51.2, 21.2, 33.7)
time <- c(496.605, 599.342, 756.101, 775.254, 882.568)

data <- t(cbind(mise, dpe, pe, time))
data <- t(apply(data, 1, function(x)(x/(max(x)))))
rownames(data) <- c("MISE", "DPE", "PE", "Time")
colnames(data) <- c(2, 4, 6, 8, 10)

colors <- c('#F59799', '#9DC7EA', '#FDC67C', '#666666')

pdf("strata.pdf", width=10, height=7)
barplot(data,
        col=colors ,
        density=c(50, 50, 50, 50),
        angle=c(45, 45, 45, -45),
        ylim=c(0, 1.5),
        beside=T, xlab="Number of Dosage Strata E",
        cex.axis=1.5, cex=2, cex.lab=2)
legend("topright", legend=rownames(data), cex=1.5,
       fill=colors, bty="o")
dev.off()