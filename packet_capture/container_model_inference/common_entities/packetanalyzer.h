#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <json/json.h>  // Assuming you're using jsoncpp for JSON handling
#include <sstream>
#include <unordered_map>
#include <list>

// Enum for Protocol
enum class Protocol {
    HTTP,
    SQL
};

// Enum for Type
enum class Type {
    IN,
    OUT
};

// NetworkOperation class
class NetworkOperation {
public:
    NetworkOperation(Protocol protocol, Type type, const Json::Value &data, int socket, const std::string &path, int seq)
        : protocol_(protocol), type_(type), data_(data), socket_(socket), path_(path), seq_(seq) {}

    // Function to convert to a string representation (similar to __repr__ and toString)
    std::string toString() const {
        std::ostringstream oss;
        oss << "NetworkOperation(Sequence_No=" << seq_
            << ", protocol=" << protocolToString()
            << ", type=" << typeToString()
            << ", data=" << data_.toStyledString()
            << ", socket=" << socket_
            << ", path='" << path_ << "')";
        return oss.str();
    }

    // Function to convert to JSON string
    std::string json_str() const {
        Json::Value dict = to_dict();
        Json::StreamWriterBuilder writer;
        return Json::writeString(writer, dict);
    }

    // Convert to dictionary (for JSON serialization)
    Json::Value to_dict() const {
        Json::Value dict;
        dict["protocol"] = protocolToString();
        dict["type"] = typeToString();
        dict["data"] = data_;
        dict["socket"] = socket_;
        dict["path"] = path_;
        dict["seq"] = seq_;
        return dict;
    }

private:
    Protocol protocol_;
    Type type_;
    Json::Value data_;
    int socket_;
    std::string path_;
    int seq_;

    // Helper functions to convert enums to strings
    std::string protocolToString() const {
        switch (protocol_) {
            case Protocol::HTTP: return "HTTP";
            case Protocol::SQL: return "SQL";
            default: return "Unknown";
        }
    }

    std::string typeToString() const {
        switch (type_) {
            case Type::IN: return "In";
            case Type::OUT: return "Out";
            default: return "Unknown";
        }
    }
};

// ContainerOpp class
class ContainerOpp {
public:
    ContainerOpp(const std::string &name) : name_(name) {}

    void addPacket(const std::string &key, const NetworkOperation &packet) {
        packet_dict_[key].push_back(packet);
        packet_list_.push_back(packet);
    }

    const std::string &getName() const { return name_; }

private:
    std::string name_;
    std::unordered_map<std::string, std::list<NetworkOperation>> packet_dict_;
    std::list<NetworkOperation> packet_list_;
};

std::string parse_http_packet(const std::string &http_packet, int soc, int seq, Type packet_type);

