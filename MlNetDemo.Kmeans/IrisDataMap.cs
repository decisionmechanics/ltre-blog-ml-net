namespace MlNetDemo.Kmeans
{
    using CsvHelper.Configuration;

    public class IrisDataMap : ClassMap<Iris>
    {
        /* Constructors */

        public IrisDataMap()
        {
            Initialize();
        }

        /* Private instance methods */

        private void Initialize()
        {
            Map(x => x.SepalLength);
            Map(x => x.SepalWidth);
            Map(x => x.PetalLength);
            Map(x => x.PetalWidth);
            Map(x => x.Type);
        }
    }
}