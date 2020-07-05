using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Body_Detection
{
    class Program
    {
        static void Main(string[] args)
        {
            if (!IsPlaformCompatable()) return;
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Run();
        }

        public static void Run()
        {
            Stopwatch watch = Stopwatch.StartNew();
            Rectangle[] regions = GetBodies("ParticipantsGO2009.jpg");
            MCvAvgComp[][] facesDetected = GetFaces("ParticipantsGO2009.jpg");

            Image<Bgr, Byte> image = new Image<Bgr, byte>("ParticipantsGO2009.jpg");
            foreach (Rectangle pedestrain in regions)
            {
                image.Draw(pedestrain, new Bgr(Color.Red), 1);
            }

            foreach (MCvAvgComp f in facesDetected[0])
            {
                //draw the face detected in the 0th (gray) channel with blue color
                image.Draw(f.rect, new Bgr(Color.Blue), 2);
            }

            //MessageBox.Show((regions.Length).ToString());
            //MessageBox.Show((facesDetected.Length).ToString());


            ImageViewer.Show(
               image,
               String.Format("Pedestrain detection using {0} in {1} milliseconds.",
                  GpuInvoke.HasCuda ? "GPU" : "CPU",
                  watch.ElapsedMilliseconds));
        }

        //public static Image<Bgr, Byte> DetectFeatures()
        //{
        //    Stopwatch watch = Stopwatch.StartNew();
        //    Rectangle[] regions = GetBodies("Untitled7.jpg");
        //    MCvAvgComp[][] facesDetected = GetFaces("Untitled7.jpg");

        //    Image<Bgr, Byte> image = new Image<Bgr, byte>("Untitled7.jpg");
        //    foreach (Rectangle pedestrain in regions)
        //    {
        //        image.Draw(pedestrain, new Bgr(Color.Red), 1);
        //    }

        //    foreach (MCvAvgComp f in facesDetected[0])
        //    {
        //        //draw the face detected in the 0th (gray) channel with blue color
        //        image.Draw(f.rect, new Bgr(Color.Blue), 2);
        //    }
        //    return image;
        //}

        // Body Function
        public static Rectangle[] GetBodies(string fileName)
        {
            Image<Bgr, Byte> image = new Image<Bgr, byte>(fileName);

            Stopwatch watch;
            Rectangle[] regions;

            //check if there is a compatible GPU to run pedestrian detection
            if (GpuInvoke.HasCuda)
            {  //this is the GPU version
                using (GpuHOGDescriptor des = new GpuHOGDescriptor())
                {
                    des.SetSVMDetector(GpuHOGDescriptor.GetDefaultPeopleDetector());

                    watch = Stopwatch.StartNew();
                    using (GpuImage<Bgr, Byte> gpuImg = new GpuImage<Bgr, byte>(image))
                    using (GpuImage<Bgra, Byte> gpuBgra = gpuImg.Convert<Bgra, Byte>())
                    {
                        regions = des.DetectMultiScale(gpuBgra);
                    }
                }
            }
            else
            {  //this is the CPU version
                using (HOGDescriptor des = new HOGDescriptor())
                {
                    des.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());

                    watch = Stopwatch.StartNew();
                    regions = des.DetectMultiScale(image);
                }
            }
            watch.Stop();

            return regions;
        }
        // end Body function


        // firs face function

        public static MCvAvgComp[][] GetFaces(string fileName)
        {
            Image<Bgr, Byte> image = new Image<Bgr, byte>("Untitled7.jpg"); //Read the files as an 8-bit Bgr image  
            Image<Gray, Byte> gray = image.Convert<Gray, Byte>(); //Convert it to Grayscale

            Stopwatch watch = Stopwatch.StartNew();
            //normalizes brightness and increases contrast of the image
            gray._EqualizeHist();

            //Read the HaarCascade objects
            HaarCascade face = new HaarCascade("haarcascade_frontalface_alt_tree.xml");
            HaarCascade eye = new HaarCascade("haarcascade_eye.xml");

            //Detect the faces  from the gray scale image and store the locations as rectangle
            //The first dimensional is the channel
            //The second dimension is the index of the rectangle in the specific channel
            MCvAvgComp[][] facesDetected = gray.DetectHaarCascade(
               face,
               1.1,
               10,
               Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
               new Size(20, 20));
            watch.Stop();
            return facesDetected;

        }

        // end face function

        /// <summary>
        /// Check if both the managed and unmanaged code are compiled for the same architecture
        /// </summary>
        /// <returns>Returns true if both the managed and unmanaged code are compiled for the same architecture</returns>
        static bool IsPlaformCompatable()
        {
            int clrBitness = Marshal.SizeOf(typeof(IntPtr)) * 8;
            if (clrBitness != CvInvoke.UnmanagedCodeBitness)
            {
                MessageBox.Show(String.Format("Platform mismatched: CLR is {0} bit, C++ code is {1} bit."
                   + " Please consider recompiling the executable with the same platform target as C++ code.",
                   clrBitness, CvInvoke.UnmanagedCodeBitness));
                return false;
            }
            return true;
        }
    }
}


