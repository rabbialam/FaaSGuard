import os

import json
import networkx as nx
import matplotlib.pyplot as plt
from model import *
from utility import *
import json
from sql_network_flow_to_json import parse_it
#from word_similarity_analyzer import *

from urllib.parse import urlparse, parse_qs

def parse_path(request_path):
    # Parse the URL
    parsed_url = urlparse(request_path)
    
    # Extract query parameters
    query_params = parse_qs(parsed_url.query)
    
    # Flatten query parameters
    query_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
    
    # Extract path segments
    path_segments = parsed_url.path.strip("/").split("/")
    
    # Handle user registration paths
    if len(path_segments) == 7 and path_segments[0] == "user" and path_segments[1] == "registration":
        return json.dumps({
            "username": path_segments[2],
            "password": path_segments[3],
            "email": path_segments[4],
            "firstname": path_segments[5],
            "lastname": path_segments[6],
        })
    
    # Handle carts merge requests
    elif len(path_segments) == 3 and path_segments[0] == "carts" and path_segments[2] == "merge":
        return json.dumps({
            "cart_id": path_segments[1],
            "sessionId": query_params.get("sessionId", "")
        })
    
    # Handle generic catalogue requests
    elif path_segments[0] == "catalogue":
        return json.dumps(query_params)
    
        # Handle address addition
    elif len(path_segments) == 8 and path_segments[0] == "addresses" and path_segments[1] == "add":
        return json.dumps({
            "address_id": path_segments[2],
            "street": path_segments[3],
            "number": path_segments[4],
            "country": path_segments[5],
            "city": path_segments[6],
            "zipcode": path_segments[7]
        })
    
    # Handle cart actions (add/delete)
    elif len(path_segments) == 4 and path_segments[0] == "cart" and path_segments[1] in ["add", "delete"]:
        return json.dumps({
            "action": path_segments[1],
            "item_id": path_segments[2],
            "user_id": path_segments[3]
        })
    
    # Handle card addition
    elif len(path_segments) == 7 and path_segments[0] == "cards" and path_segments[1] == "add":
        return json.dumps({
            "card_id": path_segments[2],
            "item_id": path_segments[3],
            "expiry_month": path_segments[4],
            "expiry_year": path_segments[5],
            "user_id": path_segments[6]
        })
    
    return None
def parse_http_packet(http_packet, soc,seq,packet_type):
    lines = http_packet.split('\r\n')
    first_line = lines[0]
    headers = {}
    body =[]
    headers_complete = False
    chunked = False

    for line in lines[1:]:
        if line == '':
            headers_complete = True
        elif headers_complete:
            if 'Transfer-Encoding' in headers and headers['Transfer-Encoding'] == 'chunked':
                chunked = True
                break  # Exit the loop to handle chunked body separately
            else:
                body.append(line)
        else:
            header_name, header_value = line.split(': ', 1)
            headers[header_name] = header_value

    if chunked:
        # Handle chunked encoding
        header_data, chunked_body = http_packet.split('\r\n\r\n', 1)

        while chunked_body:
        # Get the length of the next chunk
            length_str, chunked_body = chunked_body.split('\r\n', 1)
            chunk_length = int(length_str, 16)
            if chunk_length == 0:
                break
        # Get the chunk data
            chunk_data = chunked_body[:chunk_length]
            body.append(chunk_data)
        # Remove the chunk data and the trailing CRLF
            chunked_body = chunked_body[chunk_length + 2:]
    
        # Join all body lines (not chunked)
    body = '\n'.join(body)

    # Check if the packet is a request or a response
    if first_line.startswith(('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH')):
       
        method, path, version = first_line.split(' ')
       
        
    else:
       # packet_type = Type.IN
        try:
            version, status_code, status_message = first_line.split(' ', 2)
            path = None
        except:
            print(f"error in first line {first_line}")
            return None
        #version, status_code, status_message = first_line.split(' ', 2)
        #path = None

    # Convert body to JSON if possible
    json_body = None
    if body:
        try:
            json_body = json.loads(body)
        except json.JSONDecodeError:
            print("faced json error")
            json_body = None
    if first_line.startswith('GET') and json_body == None:

        #print(f"get request path {path}")
        json_body=parse_path(path)
        if json_body == None:
            print(f"error in parsing path {path}")
            #return None
        #print(f"json formated path {json_body}")
    
    packet = NetworkOperation(Protocol.HTTP, packet_type, json_body, soc, path,seq) 
    return packet

def parse_sql_packet(sql_packet, soc,seq,packet_type):
    json_packet = parse_it(sql_packet)
    typ = json_packet['type']
    if typ == "NOT_RELATED_TO_DATA":
        return None
    #data = json_packet['data']
    packet = NetworkOperation(Protocol.SQL, packet_type, json_packet, soc, "sql",seq) 
    return packet
    

def process_sql_packet(seq, container, soc, ascii_data,packet_type):
    sql_packet = parse_sql_packet(ascii_data,soc,seq,packet_type)
    if not sql_packet:
        return None
    sql_packet.data = flatten_json(sql_packet.data)
    container.packet_list.append(sql_packet)

    return sql_packet



def process_http_packet(seq, container, soc, ascii_data,type):
    http_packet = parse_http_packet(ascii_data,soc,seq,type)
    sock_packet_list = container.packet_dict[http_packet.socket]
    if sock_packet_list :
        p = sock_packet_list[0]
        #if p.path == '/':
         #   http_packet.type= Type.OUT
        if http_packet.path == None:
            http_packet.path = p.path
        container.packet_dict.pop(http_packet.socket)
    else:
        container.packet_dict[http_packet.socket].append(http_packet)

    #if http_packet.path == None:
     #   print("path not found")
    http_packet.data = flatten_json(http_packet.data)    
    container.packet_list.append(http_packet)
    return http_packet

import string
def process_file(file_path,name):
    
    #print(f"analyziing file {file_path}")
    lines=[]
    with open(file_path, 'r',encoding="utf8",errors="ignore") as file:
        
        for line in file:
            #line = line.encode('ascii', errors='ignore').decode('ascii')

            if len(lines) >0:
                prev_line = lines[-1]
                if line.split(':')[0] == prev_line.split(':')[0]:
                    lines[-1] = lines[-1].replace('\n','')+line.split(':')[1].replace(' ','')
                   # print(f"chunked found")
                else:
                    lines.append(line)
                
            else:           
                lines.append(line)
               
    container = ContainerOpp(name)
    seq =0
    listen_socket = -1
    for line in lines:
            
        hex_data,soc,op = extract_hex_data(line)
        
        if hex_data:
            if 'read' in op:
                type = Type.IN
                if listen_socket == -1:
                    listen_socket = soc
            else:
                type = Type.OUT
            ascii_data = hex_to_ascii(hex_data)
            packet = None
            if ascii_data:
                if is_http(ascii_data) :
                    packet = process_http_packet(seq, container, soc, ascii_data,type)
                else:
                    # for sql parsing we have to send hex data
                    #print(f"non http packet {ascii_data}")
                    packet = None
                    try:
                        packet = process_sql_packet(seq,container,soc,hex_data,type)
                    
                    except:
                       # print(f"sql parse failed for line {line}")
                       continue
                   # packet = process_sql_packet(seq,container,soc,hex_data,type)
                    if not packet:
                        continue
            if packet:
                if   packet.type == Type.OUT and packet.socket == listen_socket:
                    seq =0
                    listen_socket = -1
                else: 
                    seq +=1
            else:
                print(f"ascii conv failed hex data {line}") 
        #else:
            #print(f"no hex data found for the following line line {line}")
        #if seq>7:
         #   print("problem")  
    return container

def traverse_directory(directory):
    container_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            name  = get_folder_name(file_path)
            print(f"processing file {file_path}")
            container_list.append(process_file(file_path,name))
    return container_list

def generate_graph(container):
    first_seen  = {}
    G = nx.MultiDiGraph()
    for op in container.packet_list:
        seq = op.seq
        soc = op.socket
        G.add_node(seq, label=f"Seq {seq}\n{op.type.name} Socket {op.socket} \n {op.path}") # service name/path for input and output
        for key, value in op.data.items():
            add_edge(first_seen, G, op, seq, key, value)
    #plot_graph(G,container.name)
    return G

def add_edge(first_seen, G, op, seq, key, value):
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            if sub_value not in first_seen:
                first_seen[sub_value] = seq
            else:
                G.add_edge(first_seen[sub_value], seq, label=sub_key)
    else:
        if value not in first_seen:
            if op.type == Type.IN:
                first_seen[value] = seq
        else:
            if op.type == Type.OUT:
                G.add_edge(first_seen[value], seq, label=[key,value])
def get_key(json,target):
    for key,value in json.items():
        if value == target:
            return key
    return None

def extract_rule(graph, container):
    edges_with_labels = []  
    for u, v, _, data in graph.edges(keys=True, data=True):
        k,value = data.get('label', '')
        #print(label)
        p_path = container.packet_list[u-1].path
        c_path = container.packet_list[v-1].path
        p_key = get_key(container.packet_list[u-1].data,value)
        c_key = get_key(container.packet_list[v-1].data,value)
        edges_with_labels.append((p_path,p_key, c_path, c_key,value))
    for p_path,p_key, c_path, c_key, value in edges_with_labels:
        print(f"from path {p_path} and key '{p_key}' to path {c_path} and key '{c_key}' value: {value}")
    return edges_with_labels
# def extract_rule(graph,container):

#     for node in graph.nodes():
#         op = container.packet_list[node-1]
#         #print(op)
#         if op.type == Type.OUT:
#             in_edges = graph.in_edges(node-1,data=True)
#             for u,v,data in in_edges:
#                 #print(f"edge from {u} to {v} and data {data}")
#                 p_node = container.packet_list[u-1]
#                 key,value = data['label']
#                 p_key = get_key(p_node.data,value)

#                 print(f"from path {p_node.path} and key '{p_key}' to path {op.path} and key '{key}'")


if __name__ == "__main__":
   
    directory = 'input_data_path'
   
    #check_vec()
    
    container_list = traverse_directory(directory)

    #for container in container_list:
     #   if len(container.packet_list) >0:
            
            #graph = generate_graph(container)
            #extract_rule(graph,container)
   # file_name = 'output_data_small.txt'
    #with open(file_name,'w') as file1:
    #    print("file clear")
    output_dir = 'output_directory_path'
    for container in container_list:
        if len(container.packet_list)>0:
            file_path = os.path.join(output_dir, container.name + '.txt')

            with open(file_path,'w') as file:
                print(f"packet list length {len(container.packet_list)}")
                for p in container.packet_list:
                    str_data  =  p.json_str()
                #print(str_data )
                    file.write(str_data +'\n')
            #file.flush()




    

