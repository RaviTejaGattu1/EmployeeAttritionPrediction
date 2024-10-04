# What-if scenario: Increase salary by 10% for all employees
X_test_scenario = X_test.copy()
X_test_scenario['MonthlyIncome'] = X_test_scenario['MonthlyIncome'] * 1.10

# Predict attrition based on the new scenario
y_pred_scenario = model.predict(X_test_scenario)

# Calculate new attrition rate in the scenario
new_attrition_rate = y_pred_scenario.mean()
print(f'Attrition rate after 10% salary increase: {new_attrition_rate * 100:.2f}%')

# What-if scenario: Promote all employees
X_test_scenario['JobLevel'] = X_test_scenario['JobLevel'] + 1

# Predict attrition after promotion scenario
y_pred_scenario_promotion = model.predict(X_test_scenario)

# Calculate new attrition rate in promotion scenario
promotion_attrition_rate = y_pred_scenario_promotion.mean()
print(f'Attrition rate after promotions: {promotion_attrition_rate * 100:.2f}%')
