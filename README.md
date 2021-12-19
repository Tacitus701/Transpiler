# Transpiler


# Generate models
python create_models.py


# Create c files
python transpile_simple_model.py


# Compile c files
gcc linear.c
gcc logistic.c -im


# Execute c files
./a.out
