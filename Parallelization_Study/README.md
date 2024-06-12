# Parallelization Study of the Numerical Solution to a 2D Heat Equation

This project is a study on the parallelization techniques applied to the numerical solution of the 2D heat equation. The project explores various methods to improve computational efficiency. The report for this project was created as a joint effort.

## Project Structure

- **Report**: The detailed analysis and findings of the project are documented in a report created as a joint project.
- **Scripts**:
  - `hw4.sh`: A shell script that runs `hw4_part2.py` on the CARC (UNM Center for Advanced Research Computing).
  - `hw4_output`: The output returned from the CARC run.
  - `hw4_error`: The error output from the CARC run.
  - `hw4_part1.py`: Solves part I of the homework. Use `option = 1` on line 202 for single-thread testing or `option = 2` for larger problem size and multithreading.
  - `hw4_part2.py`: The code for part II of the homework. Use `option = 1` on line 200.
  - `check_matvec.py`: Used in `hw4_part1.py` and `hw4_part2.py` to check the matrix vector.
  - `poisson.py`: Used in `hw4_part1.py` and `hw4_part2.py` to create the CSR matrix.

## Running the Code

To run the code and obtain the results:

1. **Running on CARC**:
   - Use the `hw4.sh` script to execute `hw4_part2.py` on the CARC. Ensure you have the necessary permissions and setup to run jobs on the CARC.

2. **Executing Part I**:
   - In the `hw4_part1.py` file, set `option = 1` on line 202 for single-thread testing or `option = 2` for a larger problem size and multithreading.
   ```python
   option = 1  # Single-thread testing
   # option = 2  # Larger problem size and multithreading
3. **Executing Part II**:
   - In the hw4_part2.py file, set option = 1 on line 200.
   ```python
   option = 1
