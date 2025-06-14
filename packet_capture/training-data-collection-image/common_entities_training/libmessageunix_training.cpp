#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <dlfcn.h>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <errno.h>
#include <vector>
#include <memory>
#include <sys/syscall.h>
#include <sys/types.h>
#include <errno.h>
#include <cstdlib>

static FILE *log_fd;
static ssize_t (*real_write)(int, const void *, size_t) = NULL;



void __attribute__((constructor)) initLibrary(void) {
    pid_t pid = getpid();   
    char file_name[1024];
    int file_len = snprintf(file_name, sizeof(file_name), "/tmp/logfile-%d.txt",pid);
    log_fd = fopen(file_name, "w");
    fclose(log_fd);
    char init_message[] = "Shared library is loaded\n";
   
}

void __attribute__((destructor)) cleanupLibrary(void) {
    
}

void logMessage(int fd, const char* action, const void *buf, size_t len) {
    int type;
    socklen_t type_len = sizeof(type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_len) == -1) {
        // Not a socket, skip logging
        return;
    }
    /*if(type == AF_UNIX) {
        printf("Unix socket\n");
    } else if (type == AF_INET || type == AF_INET6){
        printf("TCP socket\n");
    } */


    printf("In %s function with data len %d\n", action, len);
    
    // Prepare to convert buffer to hex
    const unsigned char *p = (const unsigned char *)buf;
    char hex_str[len * 2 + 1]={0}; // Each byte takes 2 hex digits + 1 for null terminator
    for (size_t i = 0; i < len; i++) {
        snprintf(hex_str + i * 2, 3, "%02x", p[i]); // Write two hex chars at a time
    }
    hex_str[len * 2] = '\0';
    pid_t pid = getpid();  // Get process ID
    pid_t tid = gettid();
    char log_buffer[len * 2 + 100]={0};
    char file_name[1024];

    int file_len = snprintf(file_name, sizeof(file_name), "/tmp/logfile-%d.txt",pid);
    int log_len = snprintf(log_buffer, sizeof(log_buffer), "Intercepted %s fd %d buffer content: %s\n",
                           action, fd,hex_str);
    log_fd = fopen(file_name, "a");
    if (log_fd != nullptr) {
        //printf("in fwrite fprintf %d\n",log_fd);
        //real_write(log_fd, log_buffer, log_len);
        printf("written to file\n");
       // fprintf(log_fd,log_buffer);
       
        size_t written = fwrite(log_buffer, sizeof(char),  log_len, log_fd);
        printf("File write done written bytes %d\n",written);
        fflush(log_fd); 
        fclose(log_fd);
       
    } else {
        printf("file open failed\n");
        printf("%s", log_buffer);
    }
}

ssize_t send(int sockfd, const void *buf, size_t len, int flags) {
   // printf("in send function\n");
    static ssize_t (*real_send)(int, const void *, size_t, int) = NULL;
    if (!real_send) {
        real_send = (ssize_t (*)(int, const void *, size_t, int))dlsym(RTLD_NEXT, "send");
    }
    logMessage(sockfd, "write", buf, len);
    ssize_t result = real_send(sockfd, buf, len, flags);
    
    return result;
}

//  Name mangling creates unique names in the compiled code, which helps the linker differentiate between overloaded functions.
// For that we have to use extern C
extern "C" ssize_t writev(int fd, const struct iovec *iov, int iovcnt) {
   // printf("in writev function\n");
    static ssize_t (*real_writev)(int, const struct iovec *, int) = NULL;
    if (!real_writev) {
        real_writev = (ssize_t (*)(int, const struct iovec *, int))dlsym(RTLD_NEXT, "writev");
    }
    
    // Calculate the total length of the data
    size_t total_len = 0;
    for (int i = 0; i < iovcnt; i++) {
        total_len += iov[i].iov_len;
    }
    
    // Allocate a buffer to copy the data
    char *buffer = (char *)malloc(total_len);
    if (!buffer) {
        printf("malloc failed in writev funciton\n");
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    
    // Copy the data into the buffer
    char *ptr = buffer;
    for (int i = 0; i < iovcnt; i++) {
        memcpy(ptr, iov[i].iov_base, iov[i].iov_len);
        ptr += iov[i].iov_len;
    }
    
    // Logging function (implement this as needed)
   
    
    logMessage(fd, "write", buffer, total_len);
    
    // Clean up
    free(buffer);
    
    // Call the original writev function
    ssize_t result = real_writev(fd, iov, iovcnt);
    
    return result;
}

ssize_t sendmsg(int sockfd,const struct msghdr *msg, int flags) {
      //printf("in sendmsg function\n");
    static ssize_t (*real_sendmsg)(int, const struct msghdr *,  int) = NULL;
    if (!real_sendmsg) {
        real_sendmsg = (ssize_t (*)(int, const struct msghdr *,  int))dlsym(RTLD_NEXT, "sendmsg");
    }
     size_t len = 0;
    for (size_t i = 0; i < msg->msg_iovlen; i++) {
        len += msg->msg_iov[i].iov_len;
    }
    
    // Allocate a buffer to copy the data
    char *buf = (char *)malloc(len);
    if (!buf) {
        printf("malloc failed in sendmsg funciton\n");
        perror("malloc");
        //exit(EXIT_FAILURE);
    }
    
    // Copy the data into the buffer
    char *ptr = buf;
    for (size_t i = 0; i < msg->msg_iovlen; i++) {
        memcpy(ptr, msg->msg_iov[i].iov_base, msg->msg_iov[i].iov_len);
        ptr += msg->msg_iov[i].iov_len;
    }
    logMessage(sockfd, "write", buf, len);
    free(buf);
    ssize_t result = real_sendmsg(sockfd, msg, flags);
    
    return result;
}

ssize_t recv(int sockfd, void *buf, size_t len, int flags) {
    static ssize_t (*real_recv)(int, void *, size_t, int) = NULL;
    if (!real_recv) {
        real_recv = (ssize_t (*)(int, void *, size_t, int))dlsym(RTLD_NEXT, "recv");
    }
    ssize_t result = real_recv(sockfd, buf, len, flags);
    logMessage(sockfd, "read", buf, result);
    return result;
}

ssize_t read(int fd, void *buf, size_t count) {
    static ssize_t (*real_read)(int, void *, size_t) = NULL;
    if (!real_read) {
        real_read = (ssize_t (*)(int, void *, size_t))dlsym(RTLD_NEXT, "read");
    }
    ssize_t result = real_read(fd, buf, count);
    logMessage(fd, "read", buf, result);
    return result;
}

ssize_t write(int fd, const void *buf, size_t count) {
    if (!real_write) {
        real_write = (ssize_t (*)(int, const void *, size_t))dlsym(RTLD_NEXT, "write");
    }
    //printf("in write message with fd %d\n",fd);
    logMessage(fd, "write", buf, count);
    ssize_t result = real_write(fd, buf, count);
    
    return result;
}
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen) {
      //printf("in sendto function\n");            
    static ssize_t (*real_sendto)(int, const void *, size_t, int, const struct sockaddr *, socklen_t) = NULL;
    if (!real_sendto) {
        real_sendto = (ssize_t (*)(int, const void *, size_t, int, const struct sockaddr *, socklen_t))dlsym(RTLD_NEXT, "sendto");
    }
   
    logMessage(sockfd, "write", buf, len);
    return real_sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen) {
    static ssize_t (*real_recvfrom)(int, void *, size_t, int, struct sockaddr *, socklen_t *) = NULL;
    if (!real_recvfrom) {
        real_recvfrom = (ssize_t (*)(int, void *, size_t, int, struct sockaddr *, socklen_t *))dlsym(RTLD_NEXT, "recvfrom");
    }
    ssize_t result = real_recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
    logMessage(sockfd, "read", buf, result);
    return result;
}