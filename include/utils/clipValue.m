function val = clipValue(val, valMin, valMax)
% check if value is valid

for i = 1 : 1 : size(val(:))
	if val(i) < valMin
		val(i) = valMin;
	elseif val(i) > valMax
		val(i) = valMax;
	end
end