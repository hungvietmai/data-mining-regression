# Train 2014 → Test 2016 (time split)
python baseline_regression.py --data_dir ./data --run time --target AgrInc --train_year 2014 --test_year 2016 --log_target log1p --exclude_related_income --winsor_lower 0.01 --winsor_upper 0.99 --drop_first --onehot_geo province

# Cross-validation GroupKFold=10 (trên toàn bộ dữ liệu)
python baseline_regression.py --data_dir ./data --run cv --target AgrInc --cv 10 --log_target log1p --exclude_related_income --drop_first --onehot_geo province

# Bật chuẩn hóa số + coi Year là categorical
python baseline_regression.py --data_dir ./data --run time --target AgrInc --train_year 2014 --test_year 2016 --log_target log1p --exclude_related_income --winsor_lower 0.02 --winsor_upper 0.98 --drop_first --onehot_geo province --scale_numeric --year_as_category