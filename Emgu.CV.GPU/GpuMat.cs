﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.Util;

namespace Emgu.CV.GPU
{
   /// <summary>
   /// Similar to CvArray but use GPU for processing
   /// </summary>
   /// <typeparam name="TDepth">The type of element in the matrix</typeparam>
   public class GpuMat<TDepth> : UnmanagedObject, IEquatable<GpuMat<TDepth>>
      where TDepth : new()
   {
      internal GpuMat(IntPtr ptr)
      {
         _ptr = ptr;
      }

      /// <summary>
      /// Create a GpuMat of the specified size
      /// </summary>
      /// <param name="rows">The number of rows (height)</param>
      /// <param name="cols">The number of columns (width)</param>
      /// <param name="channels">The number of channels</param>
      public GpuMat(int rows, int cols, int channels)
      {
         _ptr = GpuInvoke.GpuMatCreate(rows, cols, CvInvoke.CV_MAKETYPE((int)CvToolbox.GetMatrixDepth(typeof(TDepth)), channels));
      }

      /// <summary>
      /// Create a GpuMat of the specified size
      /// </summary>
      /// <param name="size">The size of the GpuMat</param>
      /// <param name="channels">The number of channels</param>
      public GpuMat(Size size, int channels)
         : this(size.Height, size.Width, channels)
      {
      }

      /// <summary>
      /// Create a GpuMat from an CvArray of the same depth type
      /// </summary>
      /// <param name="arr">The CvArry to be converted to GpuMat</param>
      public GpuMat(CvArray<TDepth> arr)
      {
         _ptr = GpuInvoke.GpuMatCreateFromArr(arr);
      }

      /// <summary>
      /// Release the unmanaged memory associated with this GpuMat
      /// </summary>
      protected override void DisposeObject()
      {
         GpuInvoke.GpuMatRelease(ref _ptr);
      }

      /// <summary>
      /// Get the GpuMat size:
      /// width == number of columns, height == number of rows
      /// </summary>
      public Size Size
      {
         get { return GpuInvoke.GpuMatGetSize(_ptr); }
      }

      /// <summary>
      /// Get the number of channels in the GpuMat
      /// </summary>
      public int NumberOfChannels
      {
         get { return GpuInvoke.GpuMatGetChannels(_ptr); }
      }

      /// <summary>
      /// Pefroms blocking upload data to GpuMat
      /// </summary>
      /// <param name="arr">The CvArray to be uploaded to GpuMat</param>
      public void Upload(CvArray<TDepth> arr)
      {
         GpuInvoke.GpuMatUpload(_ptr, arr);
      }

      /// <summary>
      /// Downloads data from device to host memory. Blocking calls
      /// </summary>
      /// <param name="arr">The destination CvArray where the GpuMat data will be downloaded to.</param>
      public void Download(CvArray<TDepth> arr)
      {
         GpuInvoke.GpuMatDownload(_ptr, arr);
      }

      ///<summary> 
      ///Split current Image into an array of gray scale images where each element 
      ///in the array represent a single color channel of the original image
      ///</summary>
      ///<param name="gpuMats"> 
      ///An array of single channel GpuMat where each item
      ///in the array represent a single channel of the original GpuMat 
      ///</param>
      /// <param name="stream">Use a Stream to call the function asynchronously (non-blocking) or null to call the function synchronously (blocking).</param>
      public void SplitInto(GpuMat<TDepth>[] gpuMats, Stream stream)
      {
         Debug.Assert(NumberOfChannels == gpuMats.Length, "Number of channels does not agrees with the length of gpuMats");
         //If single channel, return a copy
         if (NumberOfChannels == 1)
         {
            if (stream == null)
               GpuInvoke.Copy(_ptr, gpuMats[0], IntPtr.Zero);
            else
               stream.Copy<TDepth>(this, gpuMats[0]);
         }

         //handle multiple channels
         Size size = Size;
         IntPtr[] ptrs = new IntPtr[gpuMats.Length];
         for (int i = 0; i < gpuMats.Length; i++)
         {
            Debug.Assert(gpuMats[i].Size == size, "Size mismatch");
            ptrs[i] = gpuMats[i].Ptr;
         }
         GCHandle handle = GCHandle.Alloc(ptrs, GCHandleType.Pinned);
         GpuInvoke.Split(_ptr, handle.AddrOfPinnedObject(), stream);
         handle.Free();
      }

      /// <summary>
      /// Makes multi-channel array out of several single-channel arrays
      /// </summary>
      ///<param name="gpuMats"> 
      ///An array of single channel GpuMat where each item
      ///in the array represent a single channel of the GpuMat 
      ///</param>
      /// <param name="stream">Use a Stream to call the function asynchronously (non-blocking) or null to call the function synchronously (blocking).</param>
      public void MergeFrom(GpuMat<TDepth>[] gpuMats, Stream stream)
      {
         Debug.Assert(NumberOfChannels == gpuMats.Length, "Number of channels does not agrees with the length of gpuMats");
         //If single channel, perform a copy
         if (NumberOfChannels == 1)
         {
            if (stream == null)
               GpuInvoke.Copy(gpuMats[0].Ptr, _ptr, IntPtr.Zero);
            else
               stream.Copy<TDepth>(gpuMats[0], this);
         }

         //handle multiple channels
         Size size = Size;
         IntPtr[] ptrs = new IntPtr[gpuMats.Length];
         for (int i = 0; i < gpuMats.Length; i++)
         {
            Debug.Assert(gpuMats[i].Size == size, "Size mismatch");
            ptrs[i] = gpuMats[i].Ptr;
         }
         GCHandle handle = GCHandle.Alloc(ptrs, GCHandleType.Pinned);
         GpuInvoke.Merge(handle.AddrOfPinnedObject(), _ptr, stream);
         handle.Free();
      }

      ///<summary> 
      ///Split current GpuMat into an array of single channel GpuMat where each element 
      ///in the array represent a single channel of the original GpuMat
      ///</summary>
      /// <param name="stream">Use a Stream to call the function asynchronously (non-blocking) or null to call the function synchronously (blocking).</param>
      ///<returns> 
      ///An array of single channel GpuMat where each element  
      ///in the array represent a single channel of the original GpuMat 
      ///</returns>
      public GpuMat<TDepth>[] Split(Stream stream)
      {
         GpuMat<TDepth>[] result = new GpuMat<TDepth>[NumberOfChannels];
         Size size = Size;
         for (int i = 0; i < result.Length; i++)
         {
            result[i] = new GpuMat<TDepth>(size, 1);
         }

         SplitInto(result, stream);
         return result;
      }

      /// <summary>
      /// Returns the min / max location and values for the image
      /// </summary>
      /// <param name="maxLocations">The maximum locations for each channel </param>
      /// <param name="maxValues">The maximum values for each channel</param>
      /// <param name="minLocations">The minimum locations for each channel</param>
      /// <param name="minValues">The minimum values for each channel</param>
      public void MinMax(out double[] minValues, out double[] maxValues, out Point[] minLocations, out Point[] maxLocations)
      {
         minValues = new double[NumberOfChannels];
         maxValues = new double[NumberOfChannels];
         minLocations = new Point[NumberOfChannels];
         maxLocations = new Point[NumberOfChannels];

         if (NumberOfChannels == 1)
         {
            GpuInvoke.MinMaxLoc(Ptr, ref minValues[0], ref maxValues[0], ref minLocations[0], ref maxLocations[0], IntPtr.Zero);
         }
         else
         {
            GpuMat<TDepth>[] channels = Split(null);
            try
            {
               for (int i = 0; i < NumberOfChannels; i++)
               {
                  GpuInvoke.MinMaxLoc(Ptr, ref minValues[i], ref maxValues[i], ref minLocations[i], ref maxLocations[i], IntPtr.Zero);
               }
            }
            finally
            {
               foreach (GpuMat<TDepth> mat in channels) mat.Dispose();
            }
         }
      }

      /// <summary>
      /// Returns true if the two GpuMat equals
      /// </summary>
      /// <param name="other">The other GpuMat to be compares with</param>
      /// <returns>True if the two GpuMat equals</returns>
      public bool Equals(GpuMat<TDepth> other)
      {
         if (NumberOfChannels != other.NumberOfChannels || Size != other.Size) return false;

         using (GpuMat<TDepth> xor = new GpuMat<TDepth>(Size, NumberOfChannels))
         {
            GpuInvoke.BitwiseXor(_ptr, other, xor, IntPtr.Zero, IntPtr.Zero);

            if (xor.NumberOfChannels == 1)
               return GpuInvoke.CountNonZero(xor) == 0;
            else
            {
               using (GpuMat<TDepth> singleChannel = xor.Reshape(1, 0))
               {
                  return GpuInvoke.CountNonZero(singleChannel) == 0;
               }
            }
         }
      }

      /// <summary>
      /// Convert this GpuMat to different depth
      /// </summary>
      /// <typeparam name="TOtherDepth">The depth type to convert to</typeparam>
      /// <param name="stream">Use a Stream to call the function asynchronously (non-blocking) or null to call the function synchronously (blocking).</param>
      /// <returns>GpuMat of different depth</returns>
      public GpuMat<TOtherDepth> Convert<TOtherDepth>(Stream stream)
         where TOtherDepth : new()
      {
         GpuMat<TOtherDepth> res = new GpuMat<TOtherDepth>(Size, NumberOfChannels);
         GpuInvoke.ConvertTo(Ptr, res.Ptr, 1.0, 0.0, stream);
         return res;
      }

      /// <summary>
      /// Changes shape of GpuMat without copying data.
      /// </summary>
      /// <param name="newCn">New number of channels. newCn = 0 means that the number of channels remains unchanged.</param>
      /// <param name="newRows">New number of rows. newRows = 0 means that the number of rows remains unchanged unless it needs to be changed according to newCn value.</param>
      /// <returns>A GpuMat of different shape</returns>
      public GpuMat<TDepth> Reshape(int newCn, int newRows)
      {
         return new GpuMat<TDepth>(GpuInvoke.Reshape(_ptr, newCn, newRows));
      }

      /// <summary>
      /// Copies scalar value to every selected element of the destination GpuMat:
      /// GpuMat(I)=value if mask(I)!=0
      /// </summary>
      /// <param name="value">Fill value</param>
      /// <param name="mask">Operation mask, 8-bit single channel GpuMat; specifies elements of destination array to be changed. Can be null if not used.</param>
      /// <param name="stream">Use a Stream to call the function asynchronously (non-blocking) or null to call the function synchronously (blocking).</param>
      public void SetTo(MCvScalar value, GpuMat<Byte> mask, Stream stream)
      {
         GpuInvoke.GpuMatSetTo(_ptr, value, mask, stream);
      }
   }
}
