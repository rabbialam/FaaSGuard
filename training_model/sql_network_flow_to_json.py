####################################################################################################
#

#
# @version 7/16/2024
####################################################################################################
import re
import sqlparse
from sql_metadata import Parser


def parse_it(hex_string: str) -> any:
    """
    This function takes a hex string, determines if the packet is a request or a response and
    then calls the relevant functions.

    :param hex_string: string representing sql data portion of a packet
    :return: parsed statement in json format as a dict, or command type
    """

    if len(hex_string) < 8 or len(hex_string) % 2 != 0:  # Catching definite invalid input
        return {"type": "NOT_RELATED_TO_DATA"}  # INVALID INPUT PACKET

    length = get_length(hex_string)

    if length > 2:  # Request packets typically have more than one byte
        sql_stmt = decode_request(hex_string)
        return sql_stmt
    else:  # The first packet in a response will always be a packet with lenght 1 byte
        if len(hex_string) > 10:  # There will always be more packets after the first one tho
            important_info = decode_response(hex_string)
            return important_info
        else:
            return {"type": "NOT_RELATED_TO_DATA"}  # COM_QUIT


def get_length(hex_string: str) -> int:
    """
    Given the next part of a packet, this function takes off the first 3 bytes, rearranges them
    from little endian form and then converts the hex number into an integer that represents the
    length of the packet.

    The byte sequence is converted from little endian by chopping it up into 2 letter
    strings, reversing the order and joining it back :

        103245 -> [(1,0), (3,2), (4,5)] -> [(4,5), (3,2), (1,0)] -> [4, 5, 3, 2, 1, 0] -> 453210

    :param hex_string: the next part of the packet
    :return: the length in # of characters form
    """
    length = list(zip(*[iter(hex_string[:6])] * 2))  # convert to be byte seperated list
    length.reverse()  # reverse list from little endian
    length = int(''.join([x for xs in length for x in xs]), 16)  # join back and convert to int
    return length * 2  # convert to be based on number of characters instead of bytes


def decode_request(hex_string: str) -> any:
    """
    This function takes in the data portion of the request packet and outputs
    either the parsed statement in json format as a dict, or command type

        The command type examples (not all) :
        COM_INIT_DB		    2   	change default db
        COM_QUERY			3   	a query
        COM_FIELD_LIST		4   	show fields
        COM_CREATE_DB	    5    	create database
        COM_DROP_DB		    6   	drop database
        COM_PREPARE	    	22  	prepare the statement
        COM_EXECUTE		    23  	execute the statement
        COM_CLOSE_STMT	    25	    close the prepared statement

    :param hex_string: string representing sql data portion of a packet
    :return: parsed statement in json format as a dict, or command type
    """

    packet_num = hex_string[6:8]  # represents what packet this is in the sequence (starts at 00)

    # represents the type OR if this is a result set packet, represents the num fields
    cmd = hex_string[8:10]

    # Take off the length, packet_num, and cmd
    hex_string = hex_string[10:]

    ascii_string = bytes.fromhex(hex_string).decode("ascii", errors="replace")  # to hex

    ascii_string = ascii_string.strip()
    # print("ACT: " + ascii_string)  # TODO uncomment for print stmts of plain text query

    if cmd == "03":
        return query_to_json(ascii_string) #{"type":'sql','data':ascii_string} #
    else:
        return {"type": "NOT_RELATED_TO_DATA"}  # command_types.get(cmd, 'UNKNOWN_COMMAND')


def decode_response(hex_string: str) -> dict:
    """
    This function takes in the data portion of the response packet and outputs a json like format
    that represents the result set

    :param hex_string: the data portion of the response packet
    :return: json like format representing the result set table
    """

    packet_num = hex_string[6:8]  # represents what packet this is in the sequence (starts at 00)

    # represents the type OR if this is a result set packet, represents the num fields
    num_fields = int(hex_string[8:10], 16)
    fields = []  # Stores the field names
    res = []  # Stores the array of dictionaries containing the data values

    # Take off the length, packet_num, and cmd
    hex_string = hex_string[10:]

    while hex_string != "":  # While there are still packets left

        # First 3 bytes represent the length of the data
        length = get_length(hex_string)
        hex_string = hex_string[6:]  # Remove the bytes used from the hex string

        # Next byte represents the packet num
        packet_num = hex_string[:2]
        hex_string = hex_string[2:]  # Remove the bytes used from the hex string

        # Then the data portion comes next and is as long as then length specified
        data = hex_string[:length]
        hex_string = hex_string[length:]  # Remove the bytes used from the hex string

        # The first info in the data is the command type (could also be length in some cases)
        cmd = data[:2]

        if num_fields > 0:  # If this is a field description packet
            data = data[2:]  # Remove the bytes used from the hex string
            fields.append(decode_field_description(data))
            num_fields -= 1
        elif cmd == "fe":
            data = data[2:]  # Remove the bytes used from the hex string
        else:
            res.append(decode_column_data(data, length, fields))

    return {"type": "RESULT SET", "table_values": res}


def decode_field_description(data: str):
    """
    This function takes in a field description packet and gets the important information such as
    the database name, table name (and table name alias), and the column name (and column name
    alias)

    This data is arranged as shown below :

        [ catalog name ] [ len db name ] [ db name ] [ len table name alias ] [ table name alias ]
        [ len table name ] [ table name ] [ len column name alias ] [ column name alias ]
        [ len column name ] [ column name]

    :param data: the next part of the hex string sequence
    :return: the column name (because this is the only thing needed)
    """
    # The first info is the catalog name
    catalog_name = data[:6]
    data = data[6:]

    length_db_name = int(data[:2], 16) * 2
    data = data[2:]  # Remove the bytes used from the hex string

    db_name = data[:length_db_name]
    db_name = bytes.fromhex(db_name).decode("ascii", errors="replace")  # Change hex to ASCII

    data = data[length_db_name:]  # Remove the bytes used from the hex string

    length_table_name = int(data[:2], 16) * 2
    data = data[2:]

    table_name = data[:length_table_name]
    table_name = bytes.fromhex(table_name).decode("ascii", errors="replace")

    data = data[length_table_name:]

    length_table_name_2 = int(data[:2], 16) * 2
    data = data[2:]

    table_name_2 = data[:length_table_name_2]
    table_name_2 = bytes.fromhex(table_name_2).decode("ascii", errors="replace")

    data = data[length_table_name_2:]

    length_column_name = int(data[:2], 16) * 2
    data = data[2:]

    column_name = data[:length_column_name]
    column_name = bytes.fromhex(column_name).decode("ascii", errors="replace")

    data = data[length_column_name:]

    length_column_name_2 = int(data[:2], 16) * 2
    data = data[2:]

    column_name_2 = data[:length_column_name_2]
    column_name_2 = bytes.fromhex(column_name_2).decode("ascii", errors="replace")

    data = data[length_column_name_2:]

    return column_name_2


def decode_column_data(data: str, length: int, fields: list) -> dict:
    """
    Decodes the data that represents each row in the table.

    This is formatted per row so for a table that looks like below :
        +--------+-----------+----------+---------------------------+------+
        | UserID | FirstName | LastName | Email                     | Age  |
        +--------+-----------+----------+---------------------------+------+
        |      1 | John      | Doe      | john.doe@example.com      |   28 |
        |      2 | Jane      | Smith    | jane.smith@example.com    |   34 |
        |      3 | Alice     | Johnson  | alice.johnson@example.com |   23 |
        +--------+-----------+----------+---------------------------+------+

    a call to this function on the first packet in the column data sequence would go through the
    first row and get the information below :
        Length: 2
        Column Value: 1
        Length: 8
        Column Value: John
        Length: 6
        Column Value: Doe
        Length: 40
        Column Value: john.doe@example.com
        Length: 4
        Column Value: 28

    :param data: the next part of the hex string
    :param length: the length of the packet to parse
    :param fields: list containing the column names
    :return: a dictionary of column names and there respective values
    """
    pointer = 0
    col_vals = []   # Holds all the column values
    while pointer < length:
        col_len = int(data[:2], 16) * 2     # Get the length of the next column value
        data = data[2:]                     # Remove the length from the hex string

        # Get the next column value and convert it to ascii
        col_value = bytes.fromhex(data[:col_len]).decode("ascii", errors="replace")
        col_vals.append(col_value)          # Add the value to the vals array
        data = data[col_len:]               # Remove the value part from the hex string

        pointer += int(col_len) + 2         # Increment the pointer

    res = {}
    for i in range(len(col_vals)):
        res.update({fields[i]: col_vals[i]})    # Update the dict

    return res


def get_where_clause(query: str) -> list[dict] | None:
    """
    This function takes a query and uses sqlparse to find the where clause. It then parses this
    where clause into a json like format
    :param query: the query
    :return: the where clause in an object format
    """

    where_clause = None

    for token in sqlparse.parse(query)[0].tokens:
        if token.ttype is None and token.value.upper().startswith("WHERE"):
            where_clause = str(token.value)[6:].strip(" \n;")
            break

    if where_clause is None:
        return None

    where_arr = []

    between_op = r'\b(\w+)\b\s+\bBETWEEN\b\s+(.*?)\s+\bAND\b\s+\b(\w+)\b'

    between_match = re.search(between_op, where_clause, re.IGNORECASE)

    if between_match:
        column, lower_val, upper_val = (between_match.group(1), between_match.group(2),
                                        between_match.group(3))
        where_arr.append({column.strip(): [lower_val.strip(), upper_val.strip()]})
        where_clause = where_clause.replace(between_match.group(), '')

    tokens = re.split(r'(\bAND\b|\bOR\b)', where_clause, flags=re.IGNORECASE)
    tokens = list(filter(lambda x: not bool(re.search(r'(\bAND\b|\bOR\b)', x,
                                                      flags=re.IGNORECASE)), tokens))

    for token in tokens:
        match = re.search(r"([a-zA-Z0-9.']*)\s(=|<>|<|>|<=|>=|LIKE|IN|\bIS\b\s+\bNOT\b|\bIS"
                          r"\b)\s([a-zA-Z0-9.%_(),\-\s']*)", token, re.IGNORECASE)
        if match:
            first, second = (match.group(1).strip("' \n"), match.group(3).strip("' \n"))
            where_arr.append({first: second})

    return where_arr


def get_values_info(query: str, columns: list[str]) -> list[dict] | None:
    """
    This function takes a query and the column names of the query and returns a list of dictionaries
    containing information about each value column

    First get only the values portion :
        ('JOHN', 'DOE', 'JOHN.DOE@EXAMPLE.COM', 28),
        ('JANE', 'SMITH', 'JANE.SMITH@EXAMPLE.COM', 34),
        ('ALICE', 'JOHNSON', 'ALICE.JOHNSON@EXAMPLE.COM', 23)

    Seperate each value into an array :
        ["\n('JOHN', 'DOE', 'JOHN.DOE@EXAMPLE.COM', 28", ",\n('JANE', 'SMITH',
        'JANE.SMITH@EXAMPLE.COM', 34", ",\n('ALICE', 'JOHNSON',
        'ALICE.JOHNSON@EXAMPLE.COM', 23", '']

    For each array :

        First strip leading and trailing whitespace, commas, and (
            'JOHN', 'DOE', 'JOHN.DOE@EXAMPLE.COM', 28

        Then split by commas
            ["'JOHN'", " 'DOE'", " 'JOHN.DOE@EXAMPLE.COM'", ' 28']

    :param query: the query
    :param columns: the columns of the query
    :return: a list of dictionaries, or None if this query isn't an insert
    """
    parsed = sqlparse.parse(query)[0]
    res = []

    if parsed.get_type() != 'INSERT':
        return None

    values = query.upper().split(" VALUES")[1]  # Get only the values portion
    values = values.split(")")  # Split each of the lists of values
    for j in range(len(values) - 1):
        vals_dict = {}
        val = values[j].strip(",\n (")
        val_arr = val.split(',')
        for i in range(len(val_arr)):
            vals_dict.update({columns[i]: val_arr[i].strip("' ")})
        res.append(vals_dict)

    return res


def query_to_json(sql: str) -> dict | str:
    """
    Takes a sql query and parses it into a json like format. This function uses sql-metadata for
    the column names, table names and value portion of an insert statement.

    :param sql: the sql query to parse
    :return: json like format of a sql query with only relevant information
    """

    # Make sure this is the type of query we want to parse
    if (not sql.upper().startswith("SELECT") and not sql.upper().startswith("INSERT") and not
                        sql.upper().startswith("UPDATE") and not sql.upper().startswith("DELETE")):
        return {"type": "NOT_RELATED_TO_DATA"}  # COM_QUERY

    parse = Parser(sql)

    res = {
        "type": sqlparse.parse(sql)[0].get_type(),
        "table_names": parse.tables,
        "column_names": parse.columns,
        "conditions": get_where_clause(sql),
        "values": get_values_info(sql, parse.columns)
    }

    return res


