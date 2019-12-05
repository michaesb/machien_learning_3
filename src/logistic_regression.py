from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0,
                            solver='lbfgs',
                            multi_class='multinomial',
                            max_iter=1000)

log_reg.fit(X_train, y_train.flatten())
prediction = log_reg.predict(X_test)