function a_ = RESHAPE(a, sz_new)

sz_old = size(a);

len_old = length(sz_old);
len_new = length(sz_new);

s_old = ones(1, len_old + 1);
for ii = len_old:-1:1
    s_old(ii) = s_old(ii + 1) * sz_old(len_old - ii + 1);
end

s_new = ones(1, len_new + 1);
for ii = len_new:-1:1
    s_new(ii) = s_new(ii + 1) * sz_new(len_new - ii + 1);
end

a_ = a;
for ii = len_old:-1:3
    a_ = permute(a_, [[2, 1],3:ii]);
    a_ = reshape(a_, [s_old(ii - 1), sz_old(len_old - ii + 3:len_old)]);
end
a_ = transpose(a_);
a_ = reshape(a_, 1, s_old(1));


for ii = 2:len_new
    a_ = reshape(a_, [sz_new(len_new - ii + 2), s_new(ii), sz_new(len_new - ii + 3:len_new)]);
    a_ = permute(a_, [[2, 1], 3:ii]);
end


end