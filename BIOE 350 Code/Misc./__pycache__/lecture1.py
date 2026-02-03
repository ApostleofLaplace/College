import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# functions come next
def square_var(x):
    return x**2

def return_two_vars(x):
    return x, x**2

def two_inputs(a,b=2):
    return a+b

# The main function is here, this is what runs first
def run_lecture1():
    # string variables can be declared like this

    a='hello world'
    print(a)  # and then printed

    # number variables can be declared like this
    x = 4

    # Booleans
    y = True

    # if/else
    if x>0:
        print('Positive')
    elif x==0:
        print('0')
    else:
        print('Negative')

    # for statements
    for i in range(5):
        print(i)

    # while statements
    while x<10:
        x=x+1
        print(x)

    # call function
    print(square_var(x))
    q,r = return_two_vars(x)
    print('q = ' + str(q) + ', r = ' + str(r))

    # call function with an optional input
    val = two_inputs(2)
    print(val)
    val = two_inputs(2,3)
    print(val)

    # call an inline function
    f = lambda x:x**2

    x_squared = f(x)
    print(x_squared)

    # lists
    lst = [1, 2, 3]
    # tuples
    tuple = (1,2,3)
    print(lst)
    print(tuple)

    # changing a list (but not a tuple!)
    print(lst)
    lst[0] = 10
    print(a)

    # casting an int in a list to a float
    print(float(lst[0]))

    # dictionaries
    person = {
        "name": "Alice",
        "age": 30,
        "city": "Portland"
    }
    for key, value in person.items():
        print(key, value)

    # Dictionary with lists of dictionaries
    course = {
        "title": "Data Science 101",
        "students": [
            {"name": "Alice", "grade": 91},
            {"name": "Bob", "grade": 85},
            {"name": "Charlie", "grade": 78}
        ]
    }

    # Example: get the second student's name
    print(course["students"][1]["name"])  # Output: Bob

    # numpy arrays
    my_array = np.array([1,2,3])
    print(my_array)

    # Creating arrays of zeros
    zero_vect = np.zeros(5)   # vector of zeros, 1x5
    zero_array = np.zeros([5,6]) # array of zeros, 5x6

    # Using np.arange() to create arrays
    my_array = np.arange(2,8,2)  # create array of [2, 4, 6]

    # dot product
    a = [1, 2, 3]
    b = [2, 2, 2]
    dot_product = np.dot(a,b)
    print(dot_product)

    # vector norm
    vector_norm = np.linalg.norm(a)
    print(vector_norm)

    # slice an array
    arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    slice_result = arr[0:2,1:3]

    # reshape an array
    arr = np.array([1,2,3,4,5,6])
    reshaped = np.reshape(arr,(2,3))
    print(reshaped)

    # Plotting
    x = np.linspace(0,2*np.pi,100)
    y = np.sin(x)
    plt.plot(x,y,'b:',label='sin')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    # A practical example
    x = np.linspace(0,2*np.pi,100)
    y = np.exp(-0.5 * x) * np.sin(2*np.pi * x)
    plt.plot(x,y)
    plt.show()
    find_peaks(y)
    # Find peaks
    peaks, _ = find_peaks(y)

    # Plot
    plt.plot(x, y)

    # Annotate peaks
    for p in peaks:
        plt.plot(x[p], y[p], "ro")  # mark the peak
        plt.text(x[p], y[p], f"({x[p]:.2f}, {y[p]:.2f})",
                 ha="left", va="bottom")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Annotated Peaks")
    plt.show()

    # In-class assignment
    x = np.linspace(-2, 6, 100)
    y = x**3 - 6*x**2 + 4*x + 12

    plt.plot(x,y)
    sig_max = np.max(y)
    plt.title('Maximum = '+ str(np.max(y)))
    plt.savefig('my-plot.png')
    plt.show()