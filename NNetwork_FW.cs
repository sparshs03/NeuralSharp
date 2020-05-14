using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using matManip;

namespace NN_Framework
{
    class Layer
    {
        public int nodes;
        public float[][] weights;
        public float[][] zsum;
        public float[][] acts;
        public float[][] zErr;

        public Layer(int nodes)
        {
            this.nodes = nodes;
        }
    }

    class NNetwork
    {
        public Layer[] layers;
        public float rate = 0.1f;
        matManip.matManip matmanip = new matManip.matManip();

        public NNetwork(Layer[] layers)
        {
            this.layers = layers;
            Random rand = new Random();

            //Creating Array of "Connections" / Weights
            //starting at 1 to ignore input layer
            for(int layNum = 1; layNum < this.layers.Length; layNum++)
            {  
                //rows
                float[][] weights = new float[this.layers[layNum - 1].nodes][];

                for(int j = 0; j < this.layers[layNum - 1].nodes; j++)
                {
                    //columns
                    weights[j] = new float[this.layers[layNum].nodes];

                    for(int k = 0; k < this.layers[layNum].nodes; k++)
                    {
                        weights[j][k] = (float)rand.NextDouble();
                    }
                    //assign the layer the initalized weight array
                    this.layers[layNum].weights = weights;
                }
            }
        }

        public float sigmoid(float x)
        {
            return 1 / (1 + (float)Math.Exp(-x));
        }
        
        public float d_sigmoid(float x)
        {
            return (float)Math.Exp(-x) / (float)Math.Pow(1 + (float)Math.Exp(-x), 2f);
        }

        public void enumLayers()
        {
            for(int i = 0; i < layers.Length; i++)
            {
                Console.WriteLine(layers[i].nodes);
            }
        }

        public void enumWeights()
        {
            //starting at one to ignore input layer
            for (int i = 1; i < this.layers.Length; i++)
            {
                Console.WriteLine("Layer " + Convert.ToString(i) + ":\n");
                for (int j = 0; j < this.layers[i].weights.Length; j++)
                {
                    Console.WriteLine(string.Join(", ", this.layers[i].weights[j]));
                }
                Console.WriteLine("------------");
            }
        }

        public float[][] feed(float[] x)
        {
            if(x.Length != this.layers[0].nodes){Console.WriteLine("Input Error!\n");return new float[0][];}

            //creating 2 dimensional array for inputs so we can use matrix multiplication later
            float[][] input = new float[1][];
            input[0] = x;

            this.layers[0].acts = input;

            //ignoring input layer
            for(int i = 1; i < this.layers.Length; i++)
            {
                this.layers[i].zsum = matmanip.matmul(this.layers[i - 1].acts, this.layers[i].weights);
                //creating array since layer object wants a 2 dimensional array
                float[][] acts = new float[1][];

                //LINQ method of applying sigmoid to each value in array
                acts[0] = this.layers[i].zsum[0].Select(sigmoid).ToArray();
                this.layers[i].acts = acts; 
            }
            

            return this.layers[this.layers.Length - 1].acts;
        }

        public void backprop(float[] y)
        {
            float[][] yArr = new float[1][];
            yArr[0] = y;

            //last layer is output layer
            float[][] totalErr = matmanip.matsub(this.layers[this.layers.Length - 1].acts, yArr);
            this.layers[this.layers.Length - 1].zErr = matmanip.elementWise(totalErr, matmanip.d_sigArr(this.layers[this.layers.Length - 1].zsum));

            //starting at second last layer nad ignoring input layer
            for (int i = this.layers.Length - 2; i > 0; i--)
            {
                this.layers[i].zErr = matmanip.elementWise(matmanip.matmul(this.layers[i + 1].zErr, matmanip.transpose(this.layers[i + 1].weights)), matmanip.d_sigArr(this.layers[i].zsum));                     
            }

            //updating the weights, ignoring input layer
            for (int i = 1; i < this.layers.Length; i++)
            {
               float[][] change = matmanip.scalarMul(matmanip.matmul(matmanip.transpose(this.layers[i].zErr), this.layers[i - 1].acts), this.rate);
               this.layers[i].weights = matmanip.matsub(this.layers[i].weights, matmanip.transpose(change));
            }
        }
    }

    class NNetwork_FW
    {
        static void Main(string[] args)
        {

            float[][] pred;

            Layer[] lays = new Layer[4];

            lays[0] = new Layer(2);
            lays[1] = new Layer(6);
            lays[2] = new Layer(2);
            lays[3] = new Layer(1);

            NNetwork xornetwork = new NNetwork(lays);
            
            for(int i = 0; i < 10000; i++)
            {
                pred = xornetwork.feed(new float[] { 0f, 0f });
                xornetwork.backprop(new float[] { 1f });
            }

            pred = xornetwork.feed(new float[] { 0f, 0f });

            foreach (var arr in pred)
            {
                Console.WriteLine(string.Join(" ", arr));
            }
 
            Console.ReadLine(); 
        }
    }
}
