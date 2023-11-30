"""
This is the main file of Linear Algebra Project, an assignment for the Web Backend learners.

You need to implement the following classes and functions, according to the given `readme.md` file.

The task is a little bit challenging for beginners, wish you good luck!

If you have any questions, please contact us.

Notice:

- Pylance is set to strict mode, which means you need to follow the type hints strictly.
- Try your best to make your code Pythonic and elegant with Object-Oriented Programming.
- You can add more classes and functions if you want, just make sure the requirements are met.
"""
from typing import Any, Callable, Iterator, overload


def check_operation_add_sub(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to check the validity of the two objects in addition and subtraction

    :param func: the function to decorate
    :return: the decorated function
    """

    def validate_len_and_type(a: Any, b: Any):
        if len(a) != len(b):
            raise ValueError("The length of the two vectors must be the same.")
        if type(a) is not type(b):
            try:
                a = a.matrix
                b = b.matrix
            except AttributeError:
                pass
            finally:
                if type(a) is not type(b):
                    raise TypeError("The two objects must be the same type.")
        if isinstance(a, Matrix) and (a.count_rows() != b.count_rows() or a.count_columns() != b.count_columns()):
            raise ValueError("The size of the two matrices must be the same.")

    def wrapper(a: Any, b: Any) -> Any:
        nonlocal func
        validate_len_and_type(a, b)
        return func(a, b)

    return wrapper


def check_type_external(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to check the validity of the two objects in addition and subtraction, but the two objects can be of
    different types.

    :param func: the function to decorate
    :return: the decorated function
    """

    def wrapper(a: Any, b: Any):
        nonlocal func
        try:
            a + b  # type: ignore  # pylint: disable=pointless-statement
            a - b  # type: ignore  # pylint: disable=pointless-statement
        except Exception:  # pylint: disable=broad-except
            return check_operation_add_sub(func)(a, b)
        else:
            return func(a, b)

    return wrapper


class Vector:
    """
    A basic Vector class

    Attributes:
        data (list[int | float | complex]): List containing the elements of the vector.

    Methods:
        __init__: Initializes the Vector object with the given elements.
        __getitem__: Returns the element at the specified index.
        __setitem__: Sets the element at the specified index to the given value.
        __len__: Returns the number of elements in the vector.
        __str__: Returns a string representation of the vector.
        __repr__: Returns a string representation of the vector that can be used to recreate the object.
        copy: Returns a copy of the vector.
        append: Appends a value to the end of the vector.
        pop: Removes and returns the last value of the vector.
        insert: Inserts a value at the specified index.
        remove: Removes and returns the value at the specified index.
    """

    data: list[int | float | complex]

    def __init__(self, *args: int | float | complex):
        self.data = list(args)

    def __getitem__(self, index: int):
        return self.data[index]

    def __setitem__(self, index: int, value: int | float | complex):
        self.data[index] = value

    def __len__(self):
        return len(self.data)

    def __str__(self) -> str:
        return ",".join([str(num) for num in self.data])

    def __repr__(self) -> str:
        return f"Vector({str(self)})"

    def __iter__(self) -> Iterator[int | float | complex]:
        # in fact in new versions of Python it can directly call __getitem__ from 0 to len(self)
        # but Pylance thinks it better to define an iterator. Make it happy.
        return iter(self.data)

    def copy(self):
        """
        Return a copy of the vector
        """
        return type(self)(*self.data)

    def append(self, value: int | float | complex):
        """
        Append a value to the end of the vector

        :param value: the value to append
        :return: None
        """
        self.data.append(value)

    def pop(self):
        """
        Pop the **last** value of the vector and return it

        :return: the last value
        """
        return self.data.pop()

    def insert(self, index: int, value: int | float | complex):
        """
        Insert a value to the vector at the given index

        :param index: the index where value is inserted
        :param value: the value to insert
        :return:  None
        """
        self.data.insert(index, value)

    def remove(self, index: int) -> int | float | complex:
        """
        Remove the value at the given index

        :param index: the index of the value to remove
        :return: the removed value
        """
        return self.data.pop(index)

    @check_operation_add_sub
    def __add__(self, other: 'Vector') -> 'Vector | None':
        return type(self)(*[self[index] + other[index] for index in range(len(self))])

    @check_operation_add_sub
    def __sub__(self, other: 'Vector') -> 'Vector | None':
        return type(self)(*[self[index] - other[index] for index in range(len(self))])


class RowVector(Vector):
    """
    A basic RowVector class that inherits from Vector

    init method is not needed as it is the same as Vector:
    """

    def __str__(self) -> str:
        return "\t".join([str(_) for _ in self.data])

    def __repr__(self) -> str:
        return f"RowVector({super().__str__()})"

    def __mul__(self, other: 'ColumnVector') -> int | float | complex:
        if len(self) != len(other):
            raise ValueError("The length of the two vectors must be the same.")
        return sum(a * b for a, b in zip(self, other))

    @property
    def matrix(self):
        """
        Return a matrix with the row vector as its only row
        """
        return Matrix(self)


class ColumnVector(Vector):
    """
    A basic ColumnVector class that inherits from Vector
    """

    def __str__(self) -> str:
        return "\n".join([str(_) for _ in self.data])

    def __repr__(self) -> str:
        return f"ColumnVector({super().__str__()})"

    @property
    def matrix(self):
        """
        Return a matrix with the column vector as its only column
        """
        return Matrix(*self.data, row_count=len(self), column_count=1)


class Matrix:
    """
    A basic Matrix class

    Args:
        *args (RowVector): Variable number of RowVector objects representing the rows of the matrix.

    Attributes:
        data (list[int|float|complex]): List containing the elements of the matrix.
        __row_count (int): Number of rows in the matrix.
        __column_count (int): Number of columns in the matrix.

    Methods:
        __init__: Initializes the Matrix object with the given rows.
        __getitem__: Returns the element at the specified index.
        __setitem__: Sets the element at the specified index to the given value.
        __len__: Returns the total number of elements in the matrix.
        count_rows: Returns the number of rows in the matrix.
        count_columns: Returns the number of columns in the matrix.
        get_columns: Returns a list of ColumnVector objects representing the specified columns.
        get_rows: Returns a list of RowVector objects representing the specified rows.
        __str__: Returns a string representation of the matrix.
        __repr__: Returns a string representation of the matrix that can be used to recreate the object.
        insert_row: Inserts a row at the specified index.
        insert_column: Inserts a column at the specified index.
        remove_row: Removes the row at the specified index and returns it.
        remove_column: Removes the column at the specified index and returns it.
        copy: Returns a copy of the matrix.
        pop_row: Removes and returns the last row of the matrix.
        pop_column: Removes and returns the last column of the matrix.
    """
    data: list[int | float | complex]
    __row_count: int = 0
    __column_count: int = 0

    @overload
    def __init__(self, *args: RowVector) -> None:
        ...

    @overload
    def __init__(self, *args: int | float | complex, row_count: int, column_count: int) -> None:
        ...

    def __init__(self, *args: RowVector | int | float | complex, **kwargs: int):
        if not kwargs:
            self.data: list[int | float | complex] = []
            self.__row_count = len(args)
            self.__column_count = 0
            for row in args:
                if not isinstance(row, RowVector):
                    raise TypeError("The arguments must be RowVector.")
                self.data.extend(row.data)
                col_count = len(row.data)
                if col_count != self.__column_count:
                    if self.__column_count == 0:
                        self.__column_count = col_count
                    else:
                        raise ValueError("The number of columns in each row must be the same.")

        elif kwargs.keys() == {'row_count', 'column_count'}:

            self.__row_count = kwargs['row_count']
            self.__column_count = kwargs['column_count']
            self.data = []
            for elem in args:
                if not isinstance(elem, (int, float, complex)):
                    raise TypeError("The arguments must be int, float or complex.")
                self.data.append(elem)
            if len(self.data) != self.__row_count * self.__column_count:
                raise ValueError("The number of elements must be equal to row_count * column_count.")

    def __getitem__(self, index: int) -> int | float | complex:
        return self.data[index]

    def __setitem__(self, index: int, value: int | float | complex):
        self.data[index] = value

    def __len__(self):
        return len(self.data)

    def count_rows(self):
        """
        Returns the number of rows in the matrix.

        :return: int: The number of rows in the matrix.
        """
        return self.__row_count

    def count_columns(self):
        """
        Returns the number of columns in the matrix.

        :return: int: The number of columns in the matrix.
        """
        return self.__column_count

    def get_columns(self, *args: int):
        """
        Return a list of ColumnVector, each ColumnVector is a column of the matrix

        :param args: the indices of the columns
        :return: a list of ColumnVector
        """
        return [ColumnVector(*self.data[_::self.__column_count]) for _ in args]

    def get_rows(self, *args: int) -> list[RowVector]:
        """
        Return a list of RowVector, each RowVector is a row of the matrix

        :param args: the indices of the rows
        :return: a list of RowVector
        """
        return [RowVector(*self.data[ind * self.__column_count: (ind + 1) * self.__column_count]) for ind in args]

    def __str__(self) -> str:
        return "\n".join([str(row) for row in self.get_rows(*range(self.__row_count))])

    def __repr__(self) -> str:
        ret = ",".join([str(_) for _ in self.data])
        return f"Matrix({ret}, row_count={self.__row_count}, column_count={self.__column_count})"

    def insert_row(self, index: int, row: RowVector):
        """
        Insert a row to the matrix at the given index

        :param index: the index where row is inserted
        :param row: the row to insert
        :return: None
        """
        left = self.data[:index * self.__column_count]
        right = self.data[index * self.__column_count:]
        self.data = left + row.data + right
        self.__row_count += 1

    def insert_column(self, index: int, column: ColumnVector):
        """
        Insert a column to the matrix at the given index

        :param index: the index where column is inserted
        :param column: the column to insert
        :return: None
        """

        for index, value in zip(range(index, len(self.data) + len(column.data), self.__row_count), column):
            self.data.insert(index, value)
        self.__column_count += 1

    def remove_row(self, index: int) -> RowVector:
        """
        Remove the row at the given index

        :param index: the index of the row to remove
        :return: the removed row
        """
        row = self.get_rows(index)
        left = self.data[:index * self.__column_count]
        right = self.data[(index + 1) * self.__column_count:]
        self.data = left + right
        self.__row_count -= 1
        return row[0]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Matrix):
            return False
        if len(self) != len(other):
            return False
        return all(self[index] == other[index] for index in range(len(self)))

    def remove_column(self, index: int) -> ColumnVector:
        """
        Remove the column at the given index

        :param index: the index of the column to remove
        :return: the removed column
        """
        column = self.get_columns(index)[0]
        self.__column_count -= 1
        for ind in range(index, len(self) - len(column), self.__column_count):
            self.data.pop(ind)
        return column

    def copy(self):
        """
        Return a copy of the matrix
        """
        return Matrix(*self.data, row_count=self.__row_count, column_count=self.__column_count)

    def pop_row(self):
        """
        Pop the **last** row of the matrix and return it

        :return: the last row
        """
        self.__row_count -= 1
        ret = self.get_rows(self.__row_count)[0]
        self.data = self.data[:-self.__column_count]
        return ret

    def pop_column(self) -> ColumnVector:
        """
        Pop the **last** column of the matrix and return it

        :return: the last column
        """
        self.__column_count -= 1
        ret = self.get_columns(self.__column_count)[0]
        for index in range(self.__column_count, len(self.data) - self.__row_count, self.__column_count):
            self.data.pop(index)
        return ret

    def append_row(self, row: RowVector) -> None:
        """
        Append a row to the end of the matrix

        :param row: the row to append
        :return: None
        """
        self.data.extend(row.data)
        self.__row_count += 1

    def append_column(self, column: ColumnVector) -> None:
        """
        Append a column to the end of the matrix

        :param column: the column to append
        :return: None
        """
        column_generator = iter(column)
        for index in range(self.__column_count, len(self.data) + len(column.data), self.__row_count):
            self.data.insert(index, next(column_generator))
            # using zip is also OK, but you should learn how to use iter and next.
            # You can refer to function insert_column, as they are similar.
        self.__column_count += 1

    @check_operation_add_sub
    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.count_rows() != other.count_rows() or self.count_columns() != other.count_columns():
            raise ValueError("The size of the two matrices must be the same.")
        ret = self.copy()
        for index in range(len(self)):
            ret[index] += other[index]
        return ret

    @check_operation_add_sub
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if self.count_rows() != other.count_rows() or self.count_columns() != other.count_columns():
            raise ValueError("The size of the two matrices must be the same.")
        ret = self.copy()
        for index in range(len(self)):
            ret[index] -= other[index]
        return ret

    def __mul__(self, other: 'Matrix') -> 'Matrix':
        if self.count_columns() != other.count_rows():
            raise ValueError(
                "The number of columns in the first matrix must be equal to the number of rows in the second matrix.")
        return Matrix(*(sum(map(lambda x, y: x * y, row, col)) for row in self.rows for col in other.columns),
                      row_count=self.count_rows(),
                      column_count=other.count_columns()
                      )

    @property
    def rows(self) -> list[RowVector]:
        """
        Return a list of RowVector, each RowVector is a row of the matrix

        :return: a list of RowVector
        """
        return self.get_rows(*range(self.__row_count))

    @property
    def columns(self) -> list[ColumnVector]:
        """
        Return a list of ColumnVector, each ColumnVector is a column of the matrix

        :return: a list of ColumnVector
        """
        return self.get_columns(*range(self.__column_count))

    def transpose(self, operate_on_self: bool = False, copy: bool = False) -> 'Matrix':
        """
        Return the transpose of the matrix

        :param operate_on_self: whether to operate directly on self or return a new matrix (default: False)
        :param copy: whether to return a copy or just self if operate_on_self is True (default: False)
        :return: the transpose of the matrix
        """
        if not operate_on_self:
            return self.copy().transpose(operate_on_self=True)
        new_data: list[int | float | complex] = []
        z = zip(*self.rows)
        for row in z:
            new_data.extend(row)
        self.data = new_data
        self.__row_count, self.__column_count = self.__column_count, self.__row_count
        return self.copy() if copy else self


@overload
def add(a: Matrix, b: Matrix) -> Matrix: ...


@overload
def subtract(a: Matrix, b: Matrix) -> Matrix: ...


@overload
def add(a: RowVector, b: RowVector) -> RowVector: ...


@overload
def subtract(a: RowVector, b: RowVector) -> RowVector: ...


@overload
def add(a: ColumnVector, b: ColumnVector) -> ColumnVector: ...


@overload
def subtract(a: ColumnVector, b: ColumnVector) -> ColumnVector: ...


@overload
def add(a: Vector, b: Vector) -> Vector: ...


@overload
def subtract(a: Vector, b: Vector) -> Vector: ...


@overload
def add(a: Any, b: Any) -> Any: ...


@overload
def subtract(a: Any, b: Any) -> Any: ...


@check_type_external
def add(a: Any | Matrix | RowVector | ColumnVector | Vector,
        b: Any | Matrix | RowVector | ColumnVector | Vector) -> Any | Matrix | RowVector | ColumnVector | Vector:
    """
    Add two vectors or matrices or other objects

    **Notice**: This function is just an example.
    In fact we should define `__add__` functions in each class, and then use `+` to add two objects.

    :param a: the first vector or matrix
    :param b: the second vector or matrix
    :return: the sum
    """
    if isinstance(a, Matrix) and isinstance(b, Matrix):
        return Matrix(*(add(a_row, b_row) for a_row, b_row in zip(a.rows, b.rows)))
    elif isinstance(a, RowVector):
        return RowVector(*[a[num] + b[num] for num in range(len(a))])
    elif isinstance(a, ColumnVector):
        return ColumnVector(*[a[num] + b[num] for num in range(len(a))])
    elif isinstance(a, Vector):
        return Vector(*[a[num] + b[num] for num in range(len(a))])
    else:
        return a + b


@check_type_external
def subtract(a: Any | Matrix | RowVector | ColumnVector | Vector,
             b: Any | Matrix | RowVector | ColumnVector | Vector) -> Any | Matrix | RowVector | ColumnVector | Vector:
    """
    Subtract two vectors or matrices or other objects

    :param a: the first vector or matrix
    :param b: the second vector or matrix
    :return: the difference
    """
    if isinstance(a, Matrix) and isinstance(b, Matrix):
        return Matrix(*(subtract(a_row, b_row) for a_row, b_row in zip(a.rows, b.rows)))
    elif isinstance(a, RowVector):
        return RowVector(*[a[num] - b[num] for num in range(len(a))])
    elif isinstance(a, ColumnVector):
        return ColumnVector(*[a[num] - b[num] for num in range(len(a))])
    elif isinstance(a, Vector):
        return Vector(*[a[num] - b[num] for num in range(len(a))])
    else:
        return a - b


@overload
def scalar_multiply(a: Matrix, num: int | float | complex) -> Matrix: ...


@overload
def scalar_multiply(a: RowVector, num: int | float | complex) -> RowVector: ...


@overload
def scalar_multiply(a: ColumnVector, num: int | float | complex) -> ColumnVector: ...


@overload
def scalar_multiply(a: Vector, num: int | float | complex) -> Vector: ...


@overload
def scalar_multiply(a: Any, num: int | float | complex) -> Any: ...


def scalar_multiply(a: Matrix | RowVector | ColumnVector | Vector | Any,
                    num: int | float | complex) -> Matrix | RowVector | ColumnVector | Vector | Any:
    """
    Multiply a vector or matrix by a scalar

    :param a: the vector or matrix
    :param num: the scalar
    :return: the product
    """
    if isinstance(a, Matrix):
        return Matrix(*(num * elem for elem in a), row_count=a.count_rows(), column_count=a.count_columns())
    elif isinstance(a, (RowVector, ColumnVector, Vector)):
        return type(a)(*[num * num for num in a])
    else:
        return a * num


def mul(a: Matrix | RowVector | ColumnVector | Any, b: Matrix | RowVector | ColumnVector | Any) -> Matrix | Any:
    """
    Multiply two vectors or matrices or other objects

    :param a: the first vector or matrix or something else
    :param b: the second vector or matrix or something else
    :return: the product
    """
    if isinstance(a, (int, float, complex)):
        return scalar_multiply(b, a)
    if isinstance(b, (int, float, complex)):
        return scalar_multiply(a, b)
    match a:
        case Matrix():
            match b:
                case Matrix():
                    return a * b
                case RowVector() | ColumnVector():
                    return a * b.matrix
                case _:
                    raise TypeError("The second argument must be Matrix, RowVector , ColumnVector or scalar when the "
                                    "first argument is Matrix.")
        case RowVector():
            match b:
                case Matrix():
                    return a.matrix * b
                case ColumnVector():
                    return a * b
                case _:
                    raise TypeError("The second argument must be Matrix, RowVector , ColumnVector or scalar when the "
                                    "first argument is RowVector.")
        case ColumnVector():
            match b:
                case Matrix():
                    return a.matrix * b
                case RowVector():
                    return a.matrix * b.matrix
                case _:
                    raise TypeError("The second argument must be Matrix, RowVector , ColumnVector or scalar when the "
                                    "first argument is ColumnVector.")
        case _:
            return a * b


#
#
# def transfer_equations_to_matrix(equations, vals):
#     return
#
#
# def solve(mat):
#     return


if __name__ == '__main__':
    # below is just an example, you may change it to whatever you want, or use it to test your code
    rows: list[RowVector] = []
    columns: list[ColumnVector] = []
    with open('vector.csv', 'r', encoding='utf-8') as f:
        for i in f:
            line = i.strip().split(',')
            # print(line)
            vec1 = RowVector(*map(eval, line))
            vec2 = ColumnVector(*map(eval, line))
            rows.append(vec1.copy())
            columns.append(vec2)
            # print(f"A Row Vector:\n{vec1}")
            vec1.remove(-1)
            # print(f"A Column Vector:\n{vec2}")
            vec2.append(0)
    print(rows)
    mat = Matrix(*rows)  # here you need to make a Matrix from the list of RowVectors
    # print(mat)
    # print(mat[0])
    # print(len(mat))
    # print(mat.count_rows())
    # print(mat.count_columns())
    # print(mat.get_columns(0, 1))
    # print(mat.get_rows(0, 1))
    # print(f"to insert two rows:\n{rows[0]}\n{rows[1]}")
    mat.insert_row(0, rows[0])
    mat.append_row(rows[1])
    # print(mat)
    # print(f"to insert a column:\n{columns[0]}")
    mat.insert_column(0, columns[0])

    # mat.remove_row(0)
    mat.remove_column(0)

    print(mat.copy())
    print(add('---', '---'))
    # print(add(mat, mat))
    # print(mat.transpose(operate_on_self=True, copy=True))

    # print(mat.transpose(operate_on_self=True, copy=False))
    print(add(mat, mat))
    # print(subtract(mat, mat.copy()))
    # print(scalar_multiply(mat, 2))
    print(mat * mat.transpose())
    print(mat.transpose() * mat)
    print(mul('-', 15))
    # with open('equations.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     equations = []
    #     vals = []
    #     for i in lines:
    #         if i.beginwith('-'):
    #             opr = '-'
    #         else:
    #             opr = '+'
    #         for j in i:
    #             num = ''
    #             key = ''
    #             if i in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'):
    #                 num += i
    #                 continue
    #             elif i == '=':
    #                 # add later into vals
    #                 break
    #             elif i in ('+', '-'):
    #                 opr = i
    #                 # add this into equations
    #                 continue
    #             else:
    #                 key += i

    # print(solve(transfer_equations_to_matrix(equations, vals)).out_latex_eqs())
