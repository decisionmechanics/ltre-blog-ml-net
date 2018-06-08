namespace MlNetDemo.Kmeans
{
    using Microsoft.ML.Runtime.Api;

    public class Observation
    {
        /* Public fields */

        [VectorType(4)]
        public float[] Features;

        /* Public static methods */

        public static Observation Create(Iris iris)
        {
            return new Observation
            {
                Features = new[]
                {
                    (float)iris.SepalLength,
                    (float)iris.SepalWidth,
                    (float)iris.PetalLength,
                    (float)iris.SepalWidth
                }
            };
        }
    }
}