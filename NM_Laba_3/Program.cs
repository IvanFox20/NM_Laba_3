using System;
using MathNet.Numerics.LinearAlgebra;

class Program
{
    static double F1(double x, double y)
    {
        return Math.Sin(x) - y;
    }

    static double F2(double x, double y)
    {
        return Math.Pow(x,2) + Math.Pow(y,2) - 4;
    }

    // Градиент/2
    private static Vector<double> Gradient2(double x, double y)
    {
        var wTranspose = Matrix<double>.Build.DenseOfArray(new double[,]
        {
            { Math.Cos(x), -1 },
            { 2 * x, 2 * y }
        });

        var f = Vector<double>.Build.Dense(new double[]
        {
            F1(x, y),
            F2(x, y)
        });

        return wTranspose.Multiply(f);
    }

    // Метод скорейшего спуска
    private static void GradientDescent(double initialX, double initialY, int maxIterations, double epsilon)
    {
        double x = initialX;
        double y = initialY;
        double x1 = -1000000000;
        double y1 = -1000000000;

        for (int k = 0; k < maxIterations; k++)
        {
            var grad = Gradient2(x, y);
            // векторы X и X1, представляющие текущую и предыдущую точки
            var X = Vector<double>.Build.Dense(new double[] { x, y });
            var X1 = Vector<double>.Build.Dense(new double[] { x - x1, y - y1 });

            // Продолжаем итерационный процесс пока
            if (X1.L1Norm() / X.L1Norm() < epsilon)
            {
                Console.WriteLine("Решение получено за " + (k + 1) + " итераций.");
                break;
            }

            // вычисляем коэффицент размера шага
            double numerator = Math.Pow(F1(x, y), 2) + Math.Pow(F2(x, y), 2);
            double denominator = grad.DotProduct(grad);

            double stepSize = numerator / denominator;

            // переопределяем "предыдущие" переменные
            x1 = x;
            y1 = y;

            // переопределяем переменные
            x = x - stepSize * grad[0];
            y = y - stepSize * grad[1];
        }

        Console.WriteLine("Решение системы: x = " + x + ", y = " + y);
    }

    public static void Main(string[] args)
    {
        // выбор начального приближения
        double initialX = 0.5;
        double initialY = 0.5;

        // максимальное количество итераций и эпсилон
        int maxIterations = 1000;
        double epsilon = 0.0001;

        // Запускаем метод
        GradientDescent(initialX, initialY, maxIterations, epsilon);
    }
}