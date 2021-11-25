using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

namespace UDP
{
    public class UDPSocket
    {
        private class State
        {
            public byte[] buffer = new byte[bufSize];
        }

        public bool Received { get; private set; } = false;
        public string Message { get; private set; } = null;

        private const int bufSize = 8 * 1024;
        private Socket socket = null;
        private State state = new State();
        private EndPoint epFrom = new IPEndPoint(IPAddress.Any, 0);
        private AsyncCallback recv = null;

        private static string GetLocalIPAddress()
        {
            var host = Dns.GetHostEntry(Dns.GetHostName());
            foreach (var ip in host.AddressList)
            {
                if (ip.AddressFamily == AddressFamily.InterNetwork)
                {
                    return ip.ToString();
                }
            }
            throw new Exception("No network adapters with an IPv4 address in the system!");
        }

        public void Receive(int port)
        {
            Close();
            try
            {
                Received = false;
                socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                socket.SetSocketOption(SocketOptionLevel.IP, SocketOptionName.ReuseAddress, true);
                socket.Bind(new IPEndPoint(IPAddress.Parse(GetLocalIPAddress()), port));
                socket.BeginReceiveFrom(state.buffer, 0, bufSize, SocketFlags.None, ref epFrom, recv = (ar) =>
                {
                    State so = (State)ar.AsyncState;
                    int bytes = socket.EndReceiveFrom(ar, ref epFrom);
                    socket.BeginReceiveFrom(so.buffer, 0, bufSize, SocketFlags.None, ref epFrom, recv, so);
                    Message = Encoding.ASCII.GetString(so.buffer, 0, bytes);
                    Debug.Log("RECV: " + Message);
                    Received = true;
                    Close();
                }, state);
            }
            catch (NullReferenceException)
            {
                Receive(port);
            }
            catch (SocketException)
            {
                Receive(port);
            }
        }

        public void Send(string address, int port, string text)
        {
            try
            {
                Socket socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                socket.Connect(IPAddress.Parse(address), port);
                byte[] data = Encoding.ASCII.GetBytes(text);
                socket.BeginSend(data, 0, data.Length, SocketFlags.None, (ar) =>
                {
                    State so = (State)ar.AsyncState;
                    int bytes = socket.EndSend(ar);
                    Debug.Log("SEND: " + text);
                    socket.Shutdown(SocketShutdown.Both);
                    socket.Close();
                }, state);
            }
            catch (NullReferenceException)
            {
                Receive(port);
            }
            catch (SocketException)
            {
                Receive(port);
            }
        }

        public void Close()
        {
            try
            {
                socket?.Shutdown(SocketShutdown.Both);
                socket?.Close();
                socket = null;
            }
            catch (SocketException) { }
            catch (ObjectDisposedException) { }
        }
    }
}