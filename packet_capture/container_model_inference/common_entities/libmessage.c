#include <stdio.h>
#include <sys/socket.h>
#include <dlfcn.h>
#include <string.h>

FILE *log_file = NULL;

void __attribute__((constructor)) initLibrary(void) {
    char filename[256];
    snprintf(filename, sizeof(filename), "/home/app/intercept_%d.log", getpid());
    
    FILE* fp = fopen(filename, "a");
    if (log_file == NULL) {
        printf("Failed to open file the file name is %s\n",filename);
        perror("Failed to open log file");
        return;
    }
    fprintf(log_file, "Shared library is loaded\n");
    fflush(log_file);  // Ensure the message is written immediately
}

void __attribute__((destructor)) cleanupLibrary(void) {
    if (log_file) {
        fclose(log_file);
    }
}

ssize_t send(int sockfd, const void *buf, size_t len, int flags) {
    static ssize_t (*real_send)(int, const void *, size_t, int) = NULL;
    if (!real_send) {
        real_send = (ssize_t (*)(int, const void *, size_t, int))dlsym(RTLD_NEXT, "send");
    }
    printf("Intercepted send on fd %d, buffer content: %.*s\n", sockfd, (int) len, (const char *)buf);

    fprintf(log_file, "Intercepted send on fd %d, buffer content: %.*s\n", sockfd, (int) len, (const char *)buf);
    fflush(log_file);  // Ensure the log is written immediately
    return real_send(sockfd, buf, len, flags);
}

ssize_t recv(int sockfd, void *buf, size_t len, int flags) {
    static ssize_t (*real_recv)(int, void *, size_t, int) = NULL;
    if (!real_recv) {
        real_recv = (ssize_t (*)(int, void *, size_t, int))dlsym(RTLD_NEXT, "recv");
    }
    ssize_t result = real_recv(sockfd, buf, len, flags);
    fprintf(log_file, "Intercepted recv on fd %d, buffer content: %.*s\n", sockfd, (int) result, (const char *)buf);
    fflush(log_file);
    return result;
}

ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen) {
    static ssize_t (*real_sendto)(int, const void *, size_t, int, const struct sockaddr *, socklen_t) = NULL;
    if (!real_sendto) {
        real_sendto = (ssize_t (*)(int, const void *, size_t, int, struct sockaddr *, socklen_t))dlsym(RTLD_NEXT, "sendto");
    }
    printf("Intercepted send on fd %d, buffer content: %.*s\n", sockfd, (int) len, (const char *)buf);

    fprintf(log_file, "Intercepted sendto on fd %d, buffer content: %.*s\n", sockfd, (int) len, (const char *)buf);
    fflush(log_file);
    return real_sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen) {
    static ssize_t (*real_recvfrom)(int, void *, size_t, int, struct sockaddr *, socklen_t *) = NULL;
    if (!real_recvfrom) {
        real_recvfrom = (ssize_t (*)(int, void *, size_t, int, struct sockaddr *, socklen_t *))dlsym(RTLD_NEXT, "recvfrom");
    }
    ssize_t result = real_recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
    fprintf(log_file, "Intercepted recvfrom on fd %d, buffer content: %.*s\n", sockfd, (int) result, (const char *)buf);
    fflush(log_file);
    return result;
}
