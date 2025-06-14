#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <json/json.h>  // Ensure jsoncpp is properly linked
#include "packetanalyzer.h"  // Assuming Protocol, Type, and NetworkOperation are declared in packetanalyzer.h

// Function to parse HTTP packets
void flattenJson(const Json::Value& json, std::string prefix, std::map<std::string, std::string>& flatJson) {
    for (auto it = json.begin(); it != json.end(); ++it) {
        std::string key = prefix.empty() ? it.key().asString() : prefix + "." + it.key().asString();

        if (it->isObject()) {
            flattenJson(*it, key, flatJson);  // Recursively flatten nested objects
        } else if (it->isArray()) {
            for (Json::ArrayIndex i = 0; i < it->size(); ++i) {
                flattenJson((*it)[i], key + "." + std::to_string(i), flatJson);
            }
        } else {
            flatJson[key] = it->asString();  // Add key-value pair to flat map
        }
    }
}

// Function to convert the JSON object to a flattened string
std::string jsonToString(const Json::Value& json) {
    std::map<std::string, std::string> flatJson;
    flattenJson(json, "", flatJson);

    std::ostringstream result;
    for (const auto& [key, value] : flatJson) {
        result << key << " " << value << " ";
    }

    return result.str();
}

std::string parse_http_packet(const std::string &http_packet, int soc, int seq, Type packet_type) {
    //std::cout<<"The http packet: "<<http_packet<<"\n";
    std::istringstream stream(http_packet);
    std::string line;
    std::getline(stream, line);  // Get the first line
    std::string first_line = line;

    std::map<std::string, std::string> headers;
    std::string body;
    bool headers_complete = false;
    bool chunked = false;

    // Process headers
    while (std::getline(stream, line) && !line.empty() && line != "\r") {
        std::string::size_type pos = line.find(": ");
        if (pos != std::string::npos) {
            std::string header_name = line.substr(0, pos);
            std::string header_value = line.substr(pos + 2);
            headers[header_name] = header_value;
            
        }
    }

    // Check if the Transfer-Encoding is chunked
    // Transfer-Encoding chunked
    if (headers.find("Transfer-Encoding") != headers.end() && headers["Transfer-Encoding"].compare("chunked")) {
        chunked = true;
        //std::cout<<"chunked encoding detected \n";

    }

    // Handle chunked encoding if necessary
    if (chunked) {
        while (std::getline(stream, line)) {
            // Parse chunk size (hexadecimal)
            int chunk_length = std::stoi(line, nullptr, 16);
            if (chunk_length == 0)  // End of chunks
                break;
            //std::cout<<"chunk_length "<<chunk_length<<"\n";
            // Read the exact chunk data based on chunk_length
            char *chunk_data = new char[chunk_length + 1];
            stream.read(chunk_data, chunk_length);
            chunk_data[chunk_length] = '\0';
            body.append(chunk_data);
            delete[] chunk_data;

            // Consume the trailing CRLF after each chunk
            stream.ignore(2);
        }
    } else {
        // Non-chunked encoding; read the rest of the stream as the body
        while (std::getline(stream, line)) {
            body += line + "\n";
        }
    }

    // JSON parsing and further processing can remain the same
    Json::Value json_body;
    Json::CharReaderBuilder reader;
    std::string errs;
    std::istringstream body_stream(body);
    if (!body.empty()) {
        if (!Json::parseFromStream(reader, body_stream, &json_body, &errs)) {
            std::cerr << "JSON parsing error: " << errs << std::endl;
        }
    }
    std::string result = jsonToString(json_body);
    //std::cout<<"After parsingt: "<<result<<"\n";
    return result;
}

