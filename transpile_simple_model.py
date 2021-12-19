import joblib
import numpy as np

linear = joblib.load("linear.joblib")
logistic = joblib.load("logistic.joblib")

linear_bias = f"{linear.intercept_[0]}"
logistic_bias = f"{logistic.intercept_[0]}"

linear_thetas = f"{linear.coef_[0]}"
linear_thetas = linear_thetas.replace('[', '{')
linear_thetas = linear_thetas.replace(']', '}')

logistic_thetas = f"{logistic.coef_[0]}"
logistic_thetas = logistic_thetas.replace('[', '{')
logistic_thetas = logistic_thetas.replace(']', '}')
logistic_thetas = logistic_thetas.replace(' ', ',')

linear_n_thetas = len(linear.coef_[0])
logistic_n_thetas = len(logistic.coef_[0])


with open("linear.c", "w+") as file:
    file.write(f"""#include <stdio.h>

float prediction(float* features, float* thetas, int n_thetas, float bias) 
{{
    float r = bias;
    for (int i = 0; i < n_thetas; i++)
    {{
        r += features[i] * thetas[i];
    }}
    return r;
}}

int main(int argc, char* argv[])
{{
    float X[1] = {{ {3.0} }};
    int n_thetas = {linear_n_thetas};
    float thetas[{linear_n_thetas}] = {linear_thetas};
    float bias = {linear_bias};
    float result = prediction(X, thetas, n_thetas, bias);
    printf("Linear result for 3.0 : %f\\n", result);
}}
""")

with open("logistic.c", "w+") as file:
    file.write(f"""#include <stdio.h>
#include <math.h>

float prediction(float* features, float* thetas, int n_thetas, float bias) 
{{
    float r = bias;
    for (int i = 0; i < n_thetas; i++)
    {{
        r += features[i] * thetas[i];
    }}
    return r;
}}

float sigmoid(float value)
{{
    return 1 / (1 + exp(-value));
}}

int main(int argc, char* argv[])
{{
    float X[2] = {{ {2, 1} }};
    int n_thetas = {logistic_n_thetas};
    float thetas[{logistic_n_thetas}] = {logistic_thetas};
    float bias = {logistic_bias};
    float result = prediction(X, thetas, n_thetas, bias);
    printf("Logistic result for [2, 1] : %d\\n", sigmoid(result) >= 0.5);
}}
""")

print("Compile with gcc linear.c and gcc logistic.c -lm")
