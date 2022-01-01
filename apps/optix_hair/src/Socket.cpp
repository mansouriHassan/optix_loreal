#undef UNICODE

#define WIN32_LEAN_AND_MEAN

#include <iostream>
#include "inc/Socket.h"

Socket* Socket::socket_server = nullptr;

Socket* Socket::getInstance()
{
    if (socket_server == nullptr) {
        socket_server = new Socket();
    }

    return socket_server;
}

int __cdecl Socket::socket_init(void)
{
    while(true) {

        while(!socket_connected) {
            // Initialize Winsock
            iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
            if (iResult != 0) {
                printf("WSAStartup failed with error: %d\n", iResult);
                return 1;
            }

            ZeroMemory(&hints, sizeof(hints));
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;
            hints.ai_protocol = IPPROTO_TCP;
            hints.ai_flags = AI_PASSIVE;

            // Resolve the server address and port
            iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
            if (iResult != 0) {
                printf("getaddrinfo failed with error: %d\n", iResult);
                WSACleanup();
                socket_connected = false;
                return 1;
            }

            // Create a SOCKET for connecting to server
            ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
            if (ListenSocket == INVALID_SOCKET) {
                printf("socket failed with error: %ld\n", WSAGetLastError());
                freeaddrinfo(result);
                WSACleanup();
                socket_connected = false;
                return 1;
            }

            // Setup the TCP listening socket
            iResult = bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
            if (iResult == SOCKET_ERROR) {
                printf("bind failed with error: %d\n", WSAGetLastError());
                freeaddrinfo(result);
                closesocket(ListenSocket);
                WSACleanup();
                socket_connected = false;
                return 1;
            }

            freeaddrinfo(result);

            iResult = listen(ListenSocket, SOMAXCONN);
            if (iResult == SOCKET_ERROR) {
                printf("listen failed with error: %d\n", WSAGetLastError());
                closesocket(ListenSocket);
                WSACleanup();
                socket_connected = false;
                exit(0);
                return 1;
            }

            // Accept a client socket
            ClientSocket = accept(ListenSocket, NULL, NULL);
            if (ClientSocket == INVALID_SOCKET) {
                printf("accept failed with error: %d\n", WSAGetLastError());
                closesocket(ListenSocket);
                WSACleanup();
                socket_connected = false;
                return 1;
            }
            else {
                socket_connected = true;
            }
        }
        // Receive until the peer shuts down the connection
        //while(socket_connected) {
            /*
            memset(recvbuf, 0, sizeof(recvbuf));
            iResult = recv(ClientSocket, recvbuf, recvbuflen, 0);
            if (iResult > 0) {
                printf("Bytes received: %d\n", iResult);
                printf("Message received: %s\n", recvbuf);
            }
            else if (iResult == 0) {
                printf("Connection closing...\n");
                socket_connected = false;
            }
            else {
                printf("recv failed with error: %d\n", WSAGetLastError());
                closesocket(ClientSocket);
                WSACleanup();
                socket_connected = false;
                return 1;
            }
            */
        //}
    }
    return 0;
}

int Socket::socket_send(const std::string& message) {
    // Echo the buffer back to the sender
    int iSendResult = 0;
    if(socket_connected) {
        if(ClientSocket != INVALID_SOCKET) {
            iSendResult = send(ClientSocket, message.c_str(), message.length(), 0);
            if (iSendResult == SOCKET_ERROR) {
                printf("send failed with error: %d\n", WSAGetLastError());
                closesocket(ClientSocket);
                WSACleanup();
                socket_connected = false;
                return 1;
            }

            printf("Bytes sent: %d\n", iSendResult);
        }
    }
    return iSendResult;
}

int send_image(char* server, int PORT, char* lfile, char* rfile)
{
    int socketDESC;
    struct sockaddr_in serverADDRESS;
    struct hostent* hostINFO;
    FILE* file_to_send;
    int ch;
    char toSEND[1];
    char remoteFILE[65536];
    int count1 = 1, count2 = 1, percent;

    hostINFO = gethostbyname("localhost");
    if (hostINFO == NULL)
    {
        printf("Problem interpreting host\n");
        return 1;
    }

    socketDESC = socket(AF_INET, SOCK_STREAM, 0);
    if (socketDESC < 0)
    {
        printf("Cannot create socket\n");
        return 1;
    }

    serverADDRESS.sin_family = AF_INET;
    serverADDRESS.sin_port = htons(PORT);
    serverADDRESS.sin_addr = *((struct in_addr*)hostINFO->h_addr);


    ZeroMemory(&(serverADDRESS.sin_zero), 8);

    if (connect(socketDESC, (struct sockaddr*)&serverADDRESS, sizeof(serverADDRESS)) < 0)
    {
        printf("Cannot connect\n");
        return 1;
    }

    file_to_send = fopen(lfile, "r");
    if (!file_to_send)
    {
        printf("Error opening file\n");
        closesocket(socketDESC);
        return 0;
    }

    else
    {
        long fileSIZE;
        fseek(file_to_send, 0, SEEK_END);
        fileSIZE = ftell(file_to_send);
        rewind(file_to_send);

        sprintf(remoteFILE, "FBEGIN:%s:%d\r\n", rfile, fileSIZE);
        send(socketDESC, remoteFILE, sizeof(remoteFILE), 0);

        percent = fileSIZE / 100;
        while ((ch = getc(file_to_send)) != EOF)
        {
            toSEND[0] = ch;
            send(socketDESC, toSEND, 1, 0);

            if (count1 == count2)
            {
                printf("33[0;0H");
                printf("\33[2J");
                printf("Filename: %s\n", lfile);
                printf("Filesize: %d Kb\n", fileSIZE / 1024);
                printf("Percent : %d%% ( %d Kb)\n", count1 / percent, count1 / 1024);
                count1 += percent;
            }
            count2++;

        }
    }
    fclose(file_to_send);
    closesocket(socketDESC);

    return 0;
}

int Socket::stop_socket()
{
    // No longer need server socket
    closesocket(ListenSocket);

    // shutdown the connection since we're done
    int iResult = shutdown(ClientSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", WSAGetLastError());
        closesocket(ClientSocket);
        WSACleanup();
        return 1;
    }

    // cleanup
    closesocket(ClientSocket);
    WSACleanup();

    return 0;
}