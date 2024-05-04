function [x, y] = BangBang_plot(x_in, P)
alpha = x_in(1:P);
alpha(P) = alpha(P) * 2;
beta = x_in(P + 1: 2 * P);
x_in_ = [];
for i = 1:P
    x_in_ = [x_in_, [beta(i), alpha(i)]];
end
x = [];
y = [];
s = 0;
for i = 1: 2 * P
    if x_in_(i) > 0
        x = [x, [s, s + x_in_(i)]];
        y = [y, [mod(i, 2), mod(i, 2)]];
    else
        x = [x, [s, s + abs(x_in_(i))]];
        y = [y, [mod(i, 2) - 2, mod(i, 2) - 2]];
    end
    s = s + abs(x_in_(i));
end
disp(y)
end