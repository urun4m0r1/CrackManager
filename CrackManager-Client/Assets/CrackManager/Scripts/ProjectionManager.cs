using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using System.Net.Sockets;
using System.Net;
using UDP;

[DisallowMultipleComponent]
public class ProjectionManager : MonoSingleton<ProjectionManager>
{
    private struct FixedProjector
    {
        public CameraImage cameraImage;
        public Vector3 position;
        public Vector3 forward;
        public Quaternion rotation;
    }

    private class Projector
    {
        public GameObject gameObject;
        private FixedProjector fixedProjector;
        private UnityEngine.Projector projector;

        public Projector(GameObject prefab, Shader shader, FixedProjector fixedProjector)
        {
            this.fixedProjector = fixedProjector;
            fixedProjector.position += Quaternion.AngleAxis(Vector3.Angle(fixedProjector.forward, Vector3.forward), Vector3.up) * new Vector3(0, 0, 0);
            gameObject = Instantiate(prefab, fixedProjector.position, fixedProjector.rotation);
            projector = gameObject.GetComponent<UnityEngine.Projector>();
            projector.material = new Material(shader);
            projector.material.SetTexture("_ShadowTex", fixedProjector.cameraImage.texture);
            projector.aspectRatio = (float)fixedProjector.cameraImage.width / (float)fixedProjector.cameraImage.height;
        }

        public void DestroyComponent()
        {
            Destroy(projector.material);
        }
    }

    [SerializeField] private UnityEvent onCaptureStart = null;
    [SerializeField] private UnityEvent onUploadEnd = null;
    [SerializeField] private UnityEvent onProjectionComplete = null;

    [SerializeField] private int layerMask = 5;
    [SerializeField] private Camera cameraObject = null;
    [SerializeField] private GameObject projectorPrefab = null;
    [SerializeField] private Shader projectorShader = null;

    public bool IsProcessBusy { get; set; } = false;
    public bool IsQRDecoded { get; private set; } = false;

    private List<Projector> projectors = new List<Projector>();
    private FixedProjector fixedProjector;
    private UDPSocket udp = new UDPSocket();
    private string hostIP = null;
    private int hostPort = 0;
    private int section = 0;
    private string fileHolo = null;
    private string fileResult = null;

    private void Awake()
    {
        onCaptureStart = onCaptureStart ?? new UnityEvent();
        onUploadEnd = onUploadEnd ?? new UnityEvent();
        onProjectionComplete = onProjectionComplete ?? new UnityEvent();
    }

    private void Update()
    {
        if (udp.Received)
        {
            if (udp.Message == "busy")
            {
                IsProcessBusy = true;
            }
            else if (udp.Message == "free")
            {
                IsProcessBusy = false;
            }
            else
            {
                string[] message = udp.Message.Split(':');
                if (message[0] == "clear")
                {
                    if (message[1] == "all")
                    {
                        ExecuteProjectors(DestroyProjector);
                    }
                    else
                    {
                        DestroyProjector(int.Parse(message[1]));
                    }
                }
                else if (message[0] == "section")
                {
                    section = int.Parse(message[1]);
                }
                else
                {
                    section = int.Parse(message[1]);
                    DownloadTexture();
                }
            }
            udp.Receive(hostPort + 1);
        }
    }

    public void CaptureSignal()
    {
        Vector3 origin = cameraObject.transform.position;
        Vector3 direction = cameraObject.transform.TransformDirection(Vector3.forward);
        if (!Physics.Raycast(origin, direction, out RaycastHit hit, Mathf.Infinity, 1 << layerMask))
        {
            ExecuteProjectors(HideProjector);
            onCaptureStart.Invoke();
            fixedProjector.position = cameraObject.transform.position;
            fixedProjector.forward = cameraObject.transform.forward;
            fixedProjector.rotation = cameraObject.transform.rotation;
            CaptureManager.Instance.StartCapture();
        }
    }

    public void DecodeQRInfo(string str)
    {
        string[] infos = str.Split(':');
        hostIP = infos[0];
        hostPort = int.Parse(infos[1]);
        fileHolo = infos[2];
        fileResult = infos[3];
        udp.Send(hostIP, hostPort, "");
        udp.Receive(hostPort + 1);
        Debug.Log(string.Format("Server connected {0}:{1}", hostIP, hostPort));
        IsQRDecoded = true;
    }

    public void UploadTexture(CameraImage cameraImage)
    {
        ExecuteProjectors(ShowProjector);
        fixedProjector.cameraImage = cameraImage;
        using (WebClient cli = new WebClient())
        {
            string filename = fileHolo.Replace('@', section.ToString()[0]);
            try
            {
                cli.UploadData(string.Format("ftp://{0}/{1}", hostIP, filename), cameraImage.texture.EncodeToJPG(100));
                udp.Send(hostIP, hostPort, string.Format("section:{0}", section));
            }
            catch (SocketException e)
            {
                Debug.Log(e.Message);
            }
        }
        onUploadEnd.Invoke();
    }

    private void DownloadTexture()
    {
        byte[] data = null;
        using (WebClient cli = new WebClient())
        {
            string filename = fileResult.Replace('@', section.ToString()[0]);
            try
            {
                data = cli.DownloadData(string.Format("ftp://{0}/{1}", hostIP, filename));
            }
            catch (SocketException e)
            {
                Debug.Log(e.Message);
            }
        }

        if (data != null)
        {
            Project(data);
        }
    }

    private void Project(byte[] data)
    {
        Texture2D texture = new Texture2D(fixedProjector.cameraImage.width, fixedProjector.cameraImage.height);
        texture.LoadImage(data);
        fixedProjector.cameraImage.texture = texture;
        fixedProjector.cameraImage.texture.wrapMode = TextureWrapMode.Clamp;
        Projector projector = new Projector(projectorPrefab, projectorShader, fixedProjector);

        if (projectors.Count == 0 || section == projectors.Count)
        {
            projectors.Add(projector);
        }
        else
        {
            projectors[section].DestroyComponent();
            projectors[section] = projector;
        }

        onProjectionComplete.Invoke();
        Debug.Log("Projection mapping complete");
    }

    private void ExecuteProjectors(Func<int, int> ExecuteProjector)
    {
        if (projectors.Count != 0)
        {
            for (int i = 0; i < projectors.Count; i++)
            {
                ExecuteProjector(i);
            }
        }
    }

    private int DestroyProjector(int i)
    {
        projectors[i].DestroyComponent();
        return 0;
    }

    private int ShowProjector(int i)
    {
        projectors[i].gameObject.SetActive(true);
        return 1;
    }

    private int HideProjector(int i)
    {
        projectors[i].gameObject.SetActive(false);
        return 2;
    }
}
