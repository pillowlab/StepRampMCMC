function lambda = safeSoftplusPower(x,p)
% Safe softplus-power function 
% Returns lambda = log(1+exp(x))^p
if nargin == 1
    p = 1.0;
end
    lambda = zeros(size(x));
    lambda(x<80) = log1p(exp(x(x<80))).^p;
    lambda(x>=80) = x(x>=80).^p;
end

