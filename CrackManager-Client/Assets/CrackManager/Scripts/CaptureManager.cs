using System.Collections;
using System.Linq;
using UnityEngine;
using UnityEngine.XR.WSA.WebCam;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;

public struct CameraImage
{
    public Texture2D texture;
    public int width;
    public int height;
}

[DisallowMultipleComponent]
public class CaptureManager : MonoSingleton<CaptureManager>
{
    struct WabCam
    {
        public int index;
        public WebCamDevice[] devices;
        public WebCamTexture texture;
    }

    struct WinCam
    {
        public PhotoCapture capture;
        public CameraParameters parameters;
        public Resolution resolution;
    }

    [SerializeField] private int webcamIndex = 0;

    private CameraImage cameraImage;
    private WinCam wincam;
    private WabCam webcam;

    private void Start()
    {
        if (!Application.isEditor && (PhotoCapture.SupportedResolutions?.Count() ?? 0) > 0)
        {
            Debug.Log("Use embedded camera for capturing");
            wincam.resolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
            wincam.parameters = new CameraParameters();
            wincam.parameters.hologramOpacity = 0.0f;
            wincam.parameters.cameraResolutionWidth = cameraImage.width = wincam.resolution.width;
            wincam.parameters.cameraResolutionHeight = cameraImage.height = wincam.resolution.height;
            wincam.parameters.pixelFormat = CapturePixelFormat.BGRA32;
        }
        else if (((webcam.devices = WebCamTexture.devices)?.Length ?? 0) > 0)
        {
            Debug.Log("Use external webcam devices for capturing");
            webcam.index = webcamIndex;
            webcam.texture = new WebCamTexture(webcam.devices[webcam.index].name);
            webcam.texture.Play();
            cameraImage.width = webcam.texture.width;
            cameraImage.height = webcam.texture.height;
        }
        else
        {
            Debug.Log("No camera devices found, exiting application");
            ApplicationAction.Exit();
        }
        cameraImage.texture = new Texture2D(cameraImage.width, cameraImage.height);
    }

    public void StartCapture() => StartCoroutine(Capture());

    private IEnumerator Capture()
    {
        if (!ProjectionManager.Instance.IsProcessBusy)
        {
            ProjectionManager.Instance.IsProcessBusy = true;
            yield return new WaitForEndOfFrame();
            Debug.Log("Start capturing");

            if (!Application.isEditor)
            {
                CaptureWinRT();
            }
            else
            {
                CaptureWebCam();
            }
        }
    }

    private void CaptureWinRT()
    {
        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject)
        {
            wincam.capture = captureObject;
            wincam.capture.StartPhotoModeAsync(wincam.parameters, delegate (PhotoCapture.PhotoCaptureResult result)
            {
                wincam.capture.TakePhotoAsync(OnCapturedPhotoToMemory);
            });
        });
    }

    private void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        photoCaptureFrame.UploadImageDataToTexture(cameraImage.texture);
        wincam.capture.StopPhotoModeAsync(OnStoppedPhotoMode);
        OnCaptured();
    }

    private void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        wincam.capture.Dispose();
        wincam.capture = null;
    }

    private void CaptureWebCam()
    {
        cameraImage.texture.SetPixels(webcam.texture.GetPixels());
        cameraImage.texture.Apply();
        OnCaptured();
    }

    private string DecodeQR(Texture2D img)
    {
        Mat imgMat = new Mat(img.height, img.width, CvType.CV_8UC4);
        Utils.texture2DToMat(img, imgMat);

        Mat grayMat = new Mat();
        Imgproc.cvtColor(imgMat, grayMat, Imgproc.COLOR_RGBA2GRAY);

        Mat points = new Mat();
        Mat straight_qrcode = new Mat();

        QRCodeDetector detector = new QRCodeDetector();

        bool result = detector.detect(grayMat, points);

        if (result)
        {
            string str = detector.decode(grayMat, points, straight_qrcode);
            return (str.Split(':').Length == 4) ? str : "";
        }
        else return "";
    }

    private void OnCaptured()
    {
        ProjectionManager.Instance.IsProcessBusy = false;
        if (ProjectionManager.Instance.IsQRDecoded)
        {
            ProjectionManager.Instance.UploadTexture(cameraImage);
        }
        else
        {
            string str = DecodeQR(cameraImage.texture);
            if (str != "")
            {
                ProjectionManager.Instance.DecodeQRInfo(str);
            }
        }
    }
}
