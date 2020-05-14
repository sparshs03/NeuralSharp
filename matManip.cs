using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace matManip
{
    class matManip
    {
        public float d_sigmoid(float x)
        {
            return (float)Math.Exp(-x) / (float)Math.Pow(1 + (float)Math.Exp(-x), 2f);
        }

        public float[][] matmul(float[][] mat1, float[][] mat2)
        {
            if (mat1[0].Length != mat2.Length)
            {
                Console.WriteLine("Columns don't match Rows!\n");
                return new float[0][];
            }

            float[][] newmat = new float[mat1.Length][];
            for (int i = 0; i < mat1.Length; i++)
            {
                newmat[i] = new float[mat2[0].Length];
            }

            for (int i = 0; i < mat1.Length; i++)
            {
                for (int z = 0; z < mat2[0].Length; z++)
                {
                    for (int j = 0; j < mat1[0].Length; j++)
                    {
                        newmat[i][z] += mat1[i][j] * mat2[j][z];
                    }
                }
            }

            return newmat;
        }

        //WORKS ONLY WHEN MATRICES ARE THE SAME
        public float[][] matsub(float[][] mat1, float[][] mat2)
        {
            if (mat1.Length != mat2.Length || mat1[0].Length != mat2[0].Length)
            {
                Console.WriteLine("Matrix rows/columns don't match!\n");
                return new float[0][];
            }

            //creating a new array because C# will change original memory space when subtracting
            float[][] result = new float[mat1.Length][];
            for(int i = 0; i < mat1.Length; i++)
            {
                result[i] = new float[mat1[0].Length];
            }

            for (int i = 0; i < mat1.Length; i++)
            {
                for (int j = 0; j < mat1[0].Length; j++)
                {
                    result[i][j] = mat1[i][j] - mat2[i][j];
                }
            }

            return result;
        }

        //WORKS ONLY WHEN MATRICES ARE THE SAME
        public float[][] matadd(float[][] mat1, float[][] mat2)
        {
            if (mat1.Length != mat2.Length || mat1[0].Length != mat2[0].Length)
            {
                Console.WriteLine("Matrix rows/columns don't match!\n");
                return new float[0][];
            }

            //creating a new array because C# will change original memory space when subtracting
            float[][] result = new float[mat1.Length][];
            for (int i = 0; i < mat1.Length; i++)
            {
                result[i] = new float[mat1[0].Length];
            }

            for (int i = 0; i < mat1.Length; i++)
            {
                for (int j = 0; j < mat1[0].Length; j++)
                {
                    result[i][j] = mat1[i][j] + mat2[i][j];
                }
            }

            return result;
        }

        //ONLY ONE DIMENSION
        public float[][] scalarMul(float[][] mat, float scalar)
        {
            //creating a new array because C# will change original memory space when subtracting
            float[][] result = new float[mat.Length][];
            for (int i = 0; i < mat.Length; i++)
            {
                result[i] = new float[mat[0].Length];
            }

            for (int i = 0; i < mat.Length; i++)
            {
                for (int j = 0; j < mat[0].Length; j++)
                {
                    result[i][j] = mat[i][j] * scalar;
                }
            }

            return result;
        }

        public float[][] elementWise(float[][] mat1, float[][] mat2)
        {
            if (mat1.Length != mat2.Length || mat1[0].Length != mat2[0].Length)
            {
                Console.WriteLine("Matrix rows/columns don't match!\n");
                return new float[0][];
            }

            //creating a new array because C# will change original memory space when subtracting
            float[][] result = new float[mat1.Length][];
            for (int i = 0; i < mat1.Length; i++)
            {
                result[i] = new float[mat1[0].Length];
            }

            for (int i = 0; i < mat1.Length; i++)
            {
                for (int j = 0; j < mat1[0].Length; j++)
                {
                    result[i][j] = mat1[i][j] * mat2[i][j];
                }
            }

            return result;
        }

        public float[][] transpose(float[][] mat)
        {
            float[][] Tmat = new float[mat[0].Length][];

            for (int i = 0; i < Tmat.Length; i++)
            {
                Tmat[i] = new float[mat.Length];
            }

            for (int i = 0; i < mat.Length; i++)
            {
                for (int j = 0; j < mat[0].Length; j++)
                {
                    Tmat[j][i] = mat[i][j];
                }
            }

            return Tmat;
        }

        public float[][] d_sigArr(float[][] mat)
        {
            //creating a new array because C# will change original memory space when subtracting
            float[][] result = new float[mat.Length][];
            for (int i = 0; i < mat.Length; i++)
            {
                result[i] = new float[mat[0].Length];
            }

            for (int i = 0; i < mat.Length; i++)
            {
                for (int j = 0; j < mat[0].Length; j++)
                {
                    result[i][j] = d_sigmoid(mat[i][j]);
                }
            }

            return result;
        }
    }
}
