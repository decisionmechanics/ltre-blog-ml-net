namespace MlNetDemo.Kmeans
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;

    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Models;
    using Microsoft.ML.Trainers;

    using CsvHelper;

    internal class Program
    {
        /* Private static methods */

        private static void Main(string[] args)
        {
            const int clusterCount = 3;

            if (args.Length != 1)
            {
                Console.WriteLine("Usage: <output file path>");

                Environment.Exit(1);
            }

            IEnumerable<Iris> data;

            using (var streamReader = new StreamReader("iris_data.csv"))
            {
                using (var csvReader = new CsvReader(streamReader))
                {
                    csvReader.Configuration.HasHeaderRecord = false;
                    csvReader.Configuration.RegisterClassMap<IrisDataMap>();

                    data = csvReader.GetRecords<Iris>().ToList();
                }
            }

            IEnumerable<Observation> observations = data.Select(Observation.Create).ToList();

            var pipeline = new LearningPipeline
            {
                CollectionDataSource.Create(observations),
                new KMeansPlusPlusClusterer { K = clusterCount, NormalizeFeatures = NormalizeOption.Yes, MaxIterations = 100 }                
            };

            PredictionModel<Observation, ClusterPrediction> model = pipeline.Train<Observation, ClusterPrediction>();

            var output = new List<string>
            {
                string.Join(",", new List<string>
                {
                    "Sepal Length",
                    "Sepal Width",
                    "Petal Length",
                    "Petal Width",
                    "Type",
                    "Cluster",
                    "Distance from cluster 1 centroid",
                    "Distance from cluster 2 centroid",
                    "Distance from cluster 3 centroid"
                })
            };

            output.AddRange(from x in data
                            let prediction = model.Predict(Observation.Create(x))
                            select $"{x.SepalLength},{x.SepalWidth},{x.PetalLength},{x.PetalWidth},{x.Type},{prediction.PredictedLabel},{string.Join(",", prediction.Score)}");

            File.WriteAllLines(args[0], output);

            data.ToList().ForEach(x =>
            {
                ClusterPrediction prediction = model.Predict(Observation.Create(x));

                Console.WriteLine($"Type {x.Type} was assigned to cluster {prediction.PredictedLabel}");
            });
        }
    }
}