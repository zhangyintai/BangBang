figure('Units', 'centimeters', 'Position', [10, 0, 15, 18]);
tiledlayout(8, 1, "TileSpacing", "tight");

hc = 3.04438;
f_alpha = @(x)hc * (1 - abs((x - 0.5) * 2).* abs((x - 0.5) * 2) .* (x - 0.5) * 2) / 2;
f_beta = @(x)(1 + abs((x - 0.5) * 2) .* ((x - 0.5) * 2 .^ (1/3))) / 2;

ax1 = nexttile;
tau = 0.32;
P = 2;
alpha0 = f_alpha(linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)) * tau;
alpha0(P) = alpha0(P) / 2;
beta0 = f_beta(linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)) * tau;
AP_P2 = [alpha0, beta0];
[AP_P2_x, AP_P2_y] = BangBang_plot(AP_P2, 2);
plot(ax1, AP_P2_x, AP_P2_y, 'Linewidth', 2);
xlim(ax1, [0, 4.5])
ylim(ax1, [-1, 1])
yticklabels(ax1, ["-H_2", "H_1", "H_2"])
yticks(ax1, [-1, 0, 1])
title(ax1, "AP N=2, \Deltat=0.32")

ax2 = nexttile;
opt_P2 = [0.569497686806119 0.192892017386209 0.336633308728385 0.340395611068835];
opt_P2 = [opt_P2(1:2:4), opt_P2(2:2:4)];
[opt_P2_x, opt_P2_y] = BangBang_plot(opt_P2, 2);
plot(ax2, opt_P2_x, opt_P2_y, 'Linewidth', 2);
xlim(ax2, [0, 4.5])
ylim(ax2, [-1, 1])
yticklabels(ax2, ["-H_2", "H_1", "H_2"])
yticks(ax2, [-1, 0, 1])
title(ax2, "BB N=2")

ax3 = nexttile;
tau = 0.37;
P = 3;
alpha0 = f_alpha(linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)) * tau;
alpha0(P) = alpha0(P) / 2;
beta0 = f_beta(linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)) * tau;
AP_P3 = [alpha0, beta0];
[AP_P3_x, AP_P3_y] = BangBang_plot(AP_P3, 3);
plot(ax3, AP_P3_x, AP_P3_y, 'Linewidth', 2);
xlim(ax3, [0, 4.5])
ylim(ax3, [-1, 1])
yticklabels(ax3, ["-H_2", "H_1", "H_2"])
yticks(ax3, [-1, 0, 1])
title(ax3, "AP N=3, \Deltat=0.37")

ax4 = nexttile;
% opt_P3 = [0.210008, 0.0487849, 0.187113, 0.125984, 0.0867699, 0.137633];
opt_P3 = [1.45417 -0.274234 0.496561  0.452291  0.302535  0.37406];
opt_P3 = [opt_P3(1:2:6), opt_P3(2:2:6)];
[opt_P3_x, opt_P3_y] = BangBang_plot(opt_P3, 3);
plot(ax4, opt_P3_x, opt_P3_y, 'Linewidth', 2);
xlim(ax4, [0, 4.5])
ylim(ax4, [-1, 1])
yticks(ax4, [-1, 0, 1])
yticklabels(ax4, ["-H_2", "H_1", "H_2"])
title(ax4, "BB N=3")

ax5 = nexttile;
tau = 0.39;
P = 4;
alpha0 = f_alpha(linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)) * tau;
alpha0(P) = alpha0(P) / 2;
disp(alpha0)
beta0 = f_beta(linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)) * tau;
AP_P4 = [alpha0, beta0];
[AP_P4_x, AP_P4_y] = BangBang_plot(AP_P4, 4);
plot(ax5, AP_P4_x, AP_P4_y, 'Linewidth', 2);
xlim(ax5, [0, 4.5])
ylim(ax5, [-1, 1])
yticklabels(ax5, ["-H_2", "H_1", "H_2"])
yticks(ax5, [-1, 0, 1])
title(ax5, "AP N=4, \Deltat=0.39")

ax6 = nexttile;
opt_P4 = [1.47096 -0.315254 0.523241 0.462795 0.446054 0.261136 0.232425 0.338651];
opt_P4 = [opt_P4(1:2:8), opt_P4(2:2:8)];
[opt_P4_x, opt_P4_y] = BangBang_plot(opt_P4, 4);
plot(ax6, opt_P4_x, opt_P4_y, 'Linewidth', 2);
xlim(ax6, [0, 4.5])
yticks(ax6, [-1, 0, 1])
ylim(ax6, [-1, 1])
yticklabels(ax6, ["-H_2", "H_1", "H_2"])
title(ax6, "BB N=4")

ax7 = nexttile;
tau = 0.36;
P = 5;
alpha0 = f_alpha(linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)) * tau;
alpha0(P) = alpha0(P) / 2;
disp(alpha0)
beta0 = f_beta(linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)) * tau;
AP_P5 = [alpha0, beta0];
[AP_P5_x, AP_P5_y] = BangBang_plot(AP_P5, 5);
plot(ax7, AP_P5_x, AP_P5_y, 'Linewidth', 2);
xlim(ax7, [0, 4.5])
ylim(ax7, [-1, 1])
yticklabels(ax7, ["-H_2", "H_1", "H_2"])
yticks(ax7, [-1, 0, 1])
title(ax7, "AP N=5, \Deltat=0.36")

ax8 = nexttile;
opt_P5 = [1.46064 -0.33358 0.529702 0.481742 0.456475 0.259067 0.266755 0.317639 0.0634666 0.235534];
opt_P5 = [opt_P5(1:2:10), opt_P5(2:2:10)];
[opt_P5_x, opt_P5_y] = BangBang_plot(opt_P5, 5);
plot(ax8, opt_P5_x, opt_P5_y, 'Linewidth', 2);
xlim(ax8, [0, 4.5])
ylim(ax8, [-1, 1])
yticks(ax8, [-1, 0, 1])
yticklabels(ax8, ["-H_2", "H_1", "H_2"])
title(ax8, "BB N=5")
xlabel("t")