using System;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using System.IO;
using System.Text;


namespace model
{
    class Program
    {
        static void Main(string[] args)
        {

            int numBernoullies = Int16.Parse(args[0]);
            float priors = float.Parse(args[1].Replace('.', ','));
            Range bernoulliesRange = new Range(numBernoullies).Named("bernolieRange");
            VariableArray<bool> bernoullies = Variable.Array<bool>(bernoulliesRange).Named("bernoullies");

            for (int i = 0; i < numBernoullies; i++)
            {
                bernoullies[i] = Variable.Bernoulli(priors);
            }

            Variable<bool> ANDOutput = Variable.Bernoulli(0.5);
            ANDOutput = Variable.AllTrue(bernoullies).Named("anyTrue");

            // The horrible way to do a probablistic OR in InferDotNet
            Variable<bool> OROutput = Variable.Bernoulli(0.5);
            var notOROutput = Variable.Array<bool>(bernoulliesRange);
            notOROutput[bernoulliesRange] = !bernoullies[bernoulliesRange];
            OROutput = !Variable.AllTrue(notOROutput).Named("orTrue");


            /********** inference **********/
            var InferenceEngine = new InferenceEngine(new ExpectationPropagation());
            InferenceEngine.NumberOfIterations = 50;

            Bernoulli bernoulliANDOutputPosterior = InferenceEngine.Infer<Bernoulli>(ANDOutput);
            Bernoulli bernoulliOROutputPosterior = InferenceEngine.Infer<Bernoulli>(OROutput);

            Console.WriteLine("AND: {0}", bernoulliANDOutputPosterior);
            Console.WriteLine("OR: {0}", bernoulliOROutputPosterior);

            var results = new StringBuilder();

            results.AppendLine("variable;mean");
            var line = string.Format("and;{0}", bernoulliANDOutputPosterior.GetMean());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("or;{0}", bernoulliOROutputPosterior.GetMean());
            results.AppendLine(line.Replace(',', '.'));


            File.WriteAllText("results.csv", results.ToString());
        }
    }
}
