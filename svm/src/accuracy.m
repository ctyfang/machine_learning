function result = measure_acc(y_hat, y)
  
  % STEP 1 - Check that number of elements match
  
  if(numel(y_hat) != numel(y))
    result = 0;
    fprintf("Number of predictions incorrect");
  endif
  
  % STEP 2 - Measure accuracy
  count = 0;
  for i = 1:numel(y)
    if(y_hat(i) == y(i))
      count += 1;
    endif
  endfor
  
  result = count / numel(y);
  
end