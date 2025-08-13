# Train 2010+2014 -> Test 2016, log target + smearing, one-hot theo huyện
python rf_regression.py --data_dir ./data --run time --target AgrInc --train_years 2010 2014 --test_year 2016 --log_target log1p --exclude_related_income --winsor_lower 0.02 --winsor_upper 0.98 --drop_first --onehot_geo district

# Cross-validation GroupKFold=10
python rf_regression.py --data_dir ./data --run cv   --target AgrInc --cv 10 --log_target log1p --exclude_related_income --drop_first --onehot_geo province