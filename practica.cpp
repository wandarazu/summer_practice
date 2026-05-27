#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <map>
#include <limits>
#include <set>
#include <iomanip>

// ===================== RANDOM =====================
class Random {
private:
    std::mt19937 gen;
public:
    Random() : gen(std::chrono::steady_clock::now().time_since_epoch().count()) {}

    int getInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

    double getDouble(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen);
    }

    bool getBool(double probability) {
        return getDouble(0.0, 1.0) < probability;
    }
};

Random rnd;

// ===================== ЗАДАЧА =====================
const int MIN_X = 9;
const int MAX_X = 14;
const int GENE_LEN = 3;

double objectiveFunction(int x) {
    return x * x + 0.1 * x - 23.0;
}

// Вспомогательные функции кодирования/декодирования (общие для всех ГА)
std::string encode(int x) {
    int v = x - MIN_X;
    std::string s;
    for (int i = GENE_LEN - 1; i >= 0; i--)
        s += ((v >> i) & 1) ? '1' : '0';
    return s;
}

int decode(const std::string& g) {
    int v = 0;
    for (char c : g) v = (v << 1) + (c - '0');
    return MIN_X + v;
}

// ===================== ИНДИВИД (основной ГА) =====================
struct Individual {
    std::string genotype;
    int phenotype;
    double fitness;

    Individual() : phenotype(0), fitness(0) {}
    Individual(const std::string& g) : genotype(g), phenotype(0), fitness(0) {}
};

// ===================== ОСНОВНОЙ ГЕНЕТИЧЕСКИЙ АЛГОРИТМ =====================
class GeneticAlgorithm {
private:
    std::vector<Individual> population;
    int popSize;
    int generations;
    double crossoverProb;
    double mutationProb;
    int initStrategy;
    int crossoverType;

    double globalBestFitness;
    int globalBestX;
    std::string globalBestGenotype;

    std::vector<double> bestHistory;
    std::vector<double> avgHistory;

    std::string intToBinary(int x) {
        return encode(x);
    }

    int binaryToInt(const std::string& str) {
        return decode(str);
    }

    int clampX(int x) {
        if (x < MIN_X) return MIN_X;
        if (x > MAX_X) return MAX_X;
        return x;
    }

    void calculateFitness(Individual& ind) {
        int x = binaryToInt(ind.genotype);
        x = clampX(x);
        ind.phenotype = x;
        ind.genotype = intToBinary(x);
        ind.fitness = objectiveFunction(x);
    }

    void calculatePopulationFitness() {
        for (auto& ind : population) calculateFitness(ind);
    }

    void initializePopulation() {
        population.clear();
        if (initStrategy == 1) {
            for (int i = 0; i < popSize; i++) {
                int x = rnd.getInt(MIN_X, MAX_X);
                population.push_back(Individual(encode(x)));
            }
        }
        else {
            for (int i = 0; i < popSize; i++) {
                int x = rnd.getInt(11, 13);
                population.push_back(Individual(encode(x)));
            }
        }
        calculatePopulationFitness();
    }

    std::vector<Individual> randomSelection() {
        std::vector<Individual> result;
        for (int i = 0; i < popSize; i++) {
            int idx = rnd.getInt(0, popSize - 1);
            result.push_back(population[idx]);
        }
        return result;
    }

    std::vector<Individual> inbreedingSelection() {
        std::vector<Individual> result;
        for (int i = 0; i < popSize / 2; i++) {
            int a = rnd.getInt(0, popSize - 1);
            int bestIdx = -1, bestDist = 1000;
            for (int b = 0; b < popSize; b++) {
                if (a == b) continue;
                int dist = 0;
                for (int k = 0; k < GENE_LEN; k++)
                    if (population[a].genotype[k] != population[b].genotype[k])
                        dist++;
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = b;
                }
            }
            result.push_back(population[a]);
            result.push_back(population[bestIdx]);
        }
        return result;
    }

    std::pair<Individual, Individual> twoPointCrossover(const Individual& p1, const Individual& p2) {
        int point1 = 1, point2 = 2;
        std::string c1 = p1.genotype.substr(0, point1) + p2.genotype.substr(point1, point2 - point1) + p1.genotype.substr(point2);
        std::string c2 = p2.genotype.substr(0, point1) + p1.genotype.substr(point1, point2 - point1) + p2.genotype.substr(point2);
        return { Individual(c1), Individual(c2) };
    }

    std::pair<Individual, Individual> cyclicCrossover(const Individual& p1, const Individual& p2) {
        std::string c1 = p1.genotype, c2 = p2.genotype;
        for (int i = 0; i < GENE_LEN; i += 2) std::swap(c1[i], c2[i]);
        return { Individual(c1), Individual(c2) };
    }

    std::pair<Individual, Individual> fibonacciCrossover(const Individual& p1, const Individual& p2) {
        int point = 1;
        std::string c1 = p1.genotype.substr(0, point) + p2.genotype.substr(point);
        std::string c2 = p2.genotype.substr(0, point) + p1.genotype.substr(point);
        return { Individual(c1), Individual(c2) };
    }

    void fibonacciMutation(Individual& ind) {
        if (!rnd.getBool(mutationProb)) return;
        int p1 = 1 % GENE_LEN, p2 = 2 % GENE_LEN;
        std::swap(ind.genotype[p1], ind.genotype[p2]);
    }

    void inversionMutation(Individual& ind) {
        if (!rnd.getBool(mutationProb)) return;
        std::reverse(ind.genotype.begin(), ind.genotype.end());
    }

    void eliteSelection(std::vector<Individual>& newPopulation) {
        calculatePopulationFitness();
        std::sort(population.begin(), population.end(),
            [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
        std::sort(newPopulation.begin(), newPopulation.end(),
            [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });

        std::vector<Individual> next;
        int eliteCount = std::max(1, popSize / 5);
        for (int i = 0; i < eliteCount; i++) next.push_back(population[i]);
        int idx = 0;
        while ((int)next.size() < popSize && idx < (int)newPopulation.size())
            next.push_back(newPopulation[idx++]);
        population = next;
    }

    void updateStatistics() {
        double generationBest = -1e9, generationAvg = 0;
        for (auto& ind : population) {
            generationAvg += ind.fitness;
            if (ind.fitness > generationBest) generationBest = ind.fitness;
            if (ind.fitness > globalBestFitness) {
                globalBestFitness = ind.fitness;
                globalBestX = ind.phenotype;
                globalBestGenotype = ind.genotype;
            }
        }
        generationAvg /= popSize;
        bestHistory.push_back(generationBest);
        avgHistory.push_back(generationAvg);
    }

public:
    GeneticAlgorithm(int ps, int gen, double cp, double mp, int initS, int crossT)
        : popSize(ps), generations(gen), crossoverProb(cp), mutationProb(mp),
        initStrategy(initS), crossoverType(crossT), globalBestFitness(-1e9) {
    }

    void run() {
        initializePopulation();
        std::cout << "\nInitial population:\n";
        for (auto& ind : population)
            std::cout << ind.genotype << " -> x=" << ind.phenotype << " f(x)=" << ind.fitness << "\n";
        updateStatistics();

        for (int gen = 1; gen <= generations; gen++) {
            std::vector<Individual> parents;
            auto r = randomSelection();
            auto i = inbreedingSelection();
            parents.insert(parents.end(), r.begin(), r.end());
            parents.insert(parents.end(), i.begin(), i.end());

            std::vector<Individual> children;
            for (int k = 0; k < popSize; k += 2) {
                Individual p1 = parents[rnd.getInt(0, parents.size() - 1)];
                Individual p2 = parents[rnd.getInt(0, parents.size() - 1)];
                Individual c1 = p1, c2 = p2;

                if (rnd.getBool(crossoverProb)) {
                    if (crossoverType == 1) {
                        auto pair = twoPointCrossover(p1, p2);
                        c1 = pair.first; c2 = pair.second;
                    }
                    else if (crossoverType == 2) {
                        auto pair = cyclicCrossover(p1, p2);
                        c1 = pair.first; c2 = pair.second;
                    }
                    else {
                        auto pair = fibonacciCrossover(p1, p2);
                        c1 = pair.first; c2 = pair.second;
                    }
                }

                fibonacciMutation(c1); inversionMutation(c1);
                fibonacciMutation(c2); inversionMutation(c2);
                calculateFitness(c1); calculateFitness(c2);

                children.push_back(c1);
                if ((int)children.size() < popSize) children.push_back(c2);
            }

            eliteSelection(children);
            calculatePopulationFitness();
            updateStatistics();

            std::cout << "Generation " << std::setw(3) << gen
                << " | Best = " << std::setw(6) << globalBestFitness
                << " | x = " << globalBestX << "\n";
        }
    }

    const std::vector<double>& getBestHistory() const { return bestHistory; }
    const std::vector<double>& getAvgHistory() const { return avgHistory; }
    double getBestFitness() const { return globalBestFitness; }
    int getBestX() const { return globalBestX; }
    std::string getBestGenotype() const { return globalBestGenotype; }
};

// ===================== ПРОСТОЙ ГА (для экспериментов) =====================
struct SimpleIndividual {
    std::string g;
    double fit;
    int x;

    SimpleIndividual() : g("000"), fit(-1e9), x(MIN_X) {}
    SimpleIndividual(std::string s) {
        g = s;
        int v = 0;
        for (char c : g) v = (v << 1) + (c - '0');
        x = MIN_X + v;
        // Коррекция недопустимых значений
        if (x < MIN_X) x = MIN_X;
        if (x > MAX_X) x = MAX_X;
        // Перекодируем, чтобы генотип соответствовал скорректированному фенотипу
        g = encode(x);
        fit = objectiveFunction(x);
    }
};

class SimpleGA {
    int popSize, genCount;
    double pc, pm;
    std::vector<SimpleIndividual> pop;
public:
    std::vector<double> bestHist, avgHist;
    double bestFit = -1e9;
    int bestX = MIN_X;

    SimpleGA(int p, int g, double c, double m)
        : popSize(p), genCount(g), pc(c), pm(m) {
    }

    void init(bool focusing) {
        pop.clear();
        for (int i = 0; i < popSize; i++) {
            int x;
            if (!focusing)
                x = rnd.getInt(MIN_X, MAX_X);
            else
                x = rnd.getInt(MIN_X, MAX_X - 1);  // исключаем 14
            pop.push_back(SimpleIndividual(encode(x)));
        }
    }

    SimpleIndividual tournament() {
        int a = rnd.getInt(0, popSize - 1);
        int b = rnd.getInt(0, popSize - 1);
        return (pop[a].fit > pop[b].fit) ? pop[a] : pop[b];
    }

    std::pair<SimpleIndividual, SimpleIndividual> crossover(const SimpleIndividual& p1,
        const SimpleIndividual& p2) {
        std::string c1 = p1.g, c2 = p2.g;
        if (rnd.getDouble(0.0, 1.0) < pc) {
            int a = rnd.getInt(0, GENE_LEN - 2);
            int b = rnd.getInt(a + 1, GENE_LEN - 1);
            c1 = p1.g.substr(0, a) + p2.g.substr(a, b - a) + p1.g.substr(b);
            c2 = p2.g.substr(0, a) + p1.g.substr(a, b - a) + p2.g.substr(b);
        }
        return { SimpleIndividual(c1), SimpleIndividual(c2) };
    }

    void mutate(SimpleIndividual& ind) {
        if (rnd.getDouble(0.0, 1.0) < pm) {
            int a = rnd.getInt(0, GENE_LEN - 1);
            int b = rnd.getInt(0, GENE_LEN - 1);
            std::swap(ind.g[a], ind.g[b]);
        }
        if (rnd.getDouble(0.0, 1.0) < pm) {
            std::reverse(ind.g.begin(), ind.g.end());
        }
        ind = SimpleIndividual(ind.g);  // конструктор исправит недопустимые значения
    }

    void step() {
        std::vector<SimpleIndividual> newPop;
        for (int i = 0; i < popSize / 2; i++) {
            SimpleIndividual p1 = tournament();
            SimpleIndividual p2 = tournament();
            auto [c1, c2] = crossover(p1, p2);
            mutate(c1);
            mutate(c2);
            newPop.push_back(c1);
            newPop.push_back(c2);
        }
        std::sort(pop.begin(), pop.end(),
            [](auto& a, auto& b) { return a.fit > b.fit; });
        newPop[0] = pop[0];
        pop = newPop;
    }

    void update() {
        double sum = 0;
        for (auto& ind : pop) {
            sum += ind.fit;
            if (ind.fit > bestFit) {
                bestFit = ind.fit;
                bestX = ind.x;
            }
        }
        bestHist.push_back(bestFit);
        avgHist.push_back(sum / popSize);
    }

    void run(bool focusing) {
        init(focusing);
        update();
        for (int i = 0; i < genCount; i++) {
            step();
            update();
        }
    }
};

// ===================== SVG (ЛЕГЕНДА ВНИЗУ, КРАСНАЯ ТОЧКА – ПЕРВЫЙ МАКСИМУМ) =====================
void saveSVG(const std::string& filename,
    const std::vector<double>& bestData,
    const std::vector<double>& avgData,
    const std::string& title) {
    if (bestData.empty()) return;

    int width = 1000, height = 600;
    int marginLeft = 90, marginRight = 40, marginTop = 60, marginBottom = 80;
    int plotWidth = width - marginLeft - marginRight;
    int plotHeight = height - marginTop - marginBottom;

    double maxVal = *std::max_element(bestData.begin(), bestData.end());
    double minVal = *std::min_element(avgData.begin(), avgData.end());
    minVal -= 5; maxVal += 5;
    double range = maxVal - minVal;

    // Индекс первого поколения, где достигнут максимум best
    size_t maxIndex = std::max_element(bestData.begin(), bestData.end()) - bestData.begin();

    int legendX = width - 200;
    int legendY = height - marginBottom - 80;

    std::ofstream svg(filename);
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

    // Заголовок
    svg << "<text x=\"" << width / 2 << "\" y=\"30\" text-anchor=\"middle\" font-size=\"22\" font-weight=\"bold\">"
        << title << "</text>\n";

    // Горизонтальные линии сетки
    for (int i = 0; i <= 10; i++) {
        double y = marginTop + i * (plotHeight / 10.0);
        svg << "<line x1=\"" << marginLeft << "\" y1=\"" << y
            << "\" x2=\"" << width - marginRight << "\" y2=\"" << y
            << "\" stroke=\"#dddddd\"/>\n";
    }

    // Оси
    svg << "<line x1=\"" << marginLeft << "\" y1=\"" << marginTop
        << "\" x2=\"" << marginLeft << "\" y2=\"" << height - marginBottom
        << "\" stroke=\"black\" stroke-width=\"2\"/>\n";
    svg << "<line x1=\"" << marginLeft << "\" y1=\"" << height - marginBottom
        << "\" x2=\"" << width - marginRight << "\" y2=\"" << height - marginBottom
        << "\" stroke=\"black\" stroke-width=\"2\"/>\n";

    // Подписи осей
    svg << "<text x=\"" << width / 2 << "\" y=\"" << height - 20
        << "\" text-anchor=\"middle\" font-size=\"16\">Generation</text>\n";
    svg << "<text x=\"30\" y=\"" << height / 2
        << "\" text-anchor=\"middle\" font-size=\"16\" transform=\"rotate(-90,30," << height / 2 << ")\">Fitness</text>\n";

    // Деления по Y
    for (int i = 0; i <= 10; i++) {
        double value = maxVal - i * range / 10.0;
        double y = marginTop + i * (plotHeight / 10.0);
        svg << "<text x=\"" << marginLeft - 10 << "\" y=\"" << y + 5
            << "\" text-anchor=\"end\" font-size=\"12\">"
            << std::fixed << std::setprecision(1) << value << "</text>\n";
    }

    // Линия best (синяя)
    for (size_t i = 1; i < bestData.size(); i++) {
        double x1 = marginLeft + (i - 1) * (plotWidth / double(bestData.size() - 1));
        double y1 = height - marginBottom - ((bestData[i - 1] - minVal) / range) * plotHeight;
        double x2 = marginLeft + i * (plotWidth / double(bestData.size() - 1));
        double y2 = height - marginBottom - ((bestData[i] - minVal) / range) * plotHeight;
        svg << "<line x1=\"" << x1 << "\" y1=\"" << y1
            << "\" x2=\"" << x2 << "\" y2=\"" << y2
            << "\" stroke=\"blue\" stroke-width=\"2\"/>\n";
    }

    // Точки best (синие, кроме той, где впервые достигнут максимум – она красная)
    for (size_t i = 0; i < bestData.size(); i++) {
        double x = marginLeft + i * (plotWidth / double(bestData.size() - 1));
        double y = height - marginBottom - ((bestData[i] - minVal) / range) * plotHeight;
        if (i == maxIndex) {
            svg << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"8\" fill=\"red\"/>\n";
        }
        else {
            svg << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"5\" fill=\"blue\"/>\n";
        }
    }

    // Линия avg (зелёная пунктирная)
    for (size_t i = 1; i < avgData.size(); i++) {
        double x1 = marginLeft + (i - 1) * (plotWidth / double(avgData.size() - 1));
        double y1 = height - marginBottom - ((avgData[i - 1] - minVal) / range) * plotHeight;
        double x2 = marginLeft + i * (plotWidth / double(avgData.size() - 1));
        double y2 = height - marginBottom - ((avgData[i] - minVal) / range) * plotHeight;
        svg << "<line x1=\"" << x1 << "\" y1=\"" << y1
            << "\" x2=\"" << x2 << "\" y2=\"" << y2
            << "\" stroke=\"green\" stroke-width=\"2\" stroke-dasharray=\"5,5\"/>\n";
    }

    // Точки avg (зелёные)
    for (size_t i = 0; i < avgData.size(); i++) {
        double x = marginLeft + i * (plotWidth / double(avgData.size() - 1));
        double y = height - marginBottom - ((avgData[i] - minVal) / range) * plotHeight;
        svg << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"4\" fill=\"green\"/>\n";
    }

    // Легенда (внизу справа)
    svg << "<rect x=\"" << legendX << "\" y=\"" << legendY
        << "\" width=\"170\" height=\"70\" fill=\"white\" stroke=\"black\"/>\n";
    svg << "<circle cx=\"" << legendX + 20 << "\" cy=\"" << legendY + 25 << "\" r=\"5\" fill=\"blue\"/>\n";
    svg << "<text x=\"" << legendX + 35 << "\" y=\"" << legendY + 30 << "\" font-size=\"13\">Best fitness</text>\n";
    svg << "<circle cx=\"" << legendX + 20 << "\" cy=\"" << legendY + 50 << "\" r=\"5\" fill=\"green\"/>\n";
    svg << "<text x=\"" << legendX + 35 << "\" y=\"" << legendY + 55 << "\" font-size=\"13\">Average fitness</text>\n";

    svg << "</svg>";
    svg.close();
}

// ===================== ВВОД С ПРОВЕРКОЙ =====================
template<typename T>
T inputValue(const std::string& text, T minVal, T maxVal) {
    T value;
    while (true) {
        std::cout << text;
        std::cin >> value;

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number in ["
                << minVal << ", " << maxVal << "].\n";
            continue;
        }

        if (value < minVal || value > maxVal) {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Value out of range. Please enter a number in ["
                << minVal << ", " << maxVal << "].\n";
            continue;
        }

        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return value;
    }
}

// ===================== MAIN =====================
int main() {
    std::cout << "=====================================\n";
    std::cout << "GENETIC ALGORITHM - VARIANT 8\n";
    std::cout << "f(x) = x^2 + 0.1x - 23\n";
    std::cout << "x in [9,14]\n";
    std::cout << "=====================================\n";

    int popSize = inputValue("Population size (4-50): ", 4, 50);
    int generations = inputValue("Generations (10-200): ", 10, 200);
    double crossProb = inputValue("Crossover probability (0-1): ", 0.0, 1.0);
    double mutProb = inputValue("Mutation probability (0-0.5): ", 0.0, 0.5);
    int initStrat = inputValue("Init strategy (1-shotgun, 2-focus): ", 1, 2);
    int crossType = inputValue("Crossover (1-two point, 2-cyclic, 3-fibonacci): ", 1, 3);

    GeneticAlgorithm ga(popSize, generations, crossProb, mutProb, initStrat, crossType);
    ga.run();

    std::cout << "\n=====================================\n";
    std::cout << "FINAL RESULT\n";
    std::cout << "Best x = " << ga.getBestX() << "\n";
    std::cout << "Best fitness = " << ga.getBestFitness() << "\n";
    std::cout << "Best genotype = " << ga.getBestGenotype() << "\n";

    saveSVG("convergence.svg", ga.getBestHistory(), ga.getAvgHistory(),
        "Genetic Algorithm Convergence (User)");
    std::cout << "User graph saved to convergence.svg\n";

    std::cout << "\n=====================================\n";
    std::cout << "RUNNING 5 PRE-DEFINED EXPERIMENTS\n";
    std::cout << "=====================================\n";

    auto runSimpleExp = [](int pop, double pc, double pm, bool focusing, const std::string& file) {
        SimpleGA sga(pop, 60, pc, pm);
        sga.run(focusing);
        saveSVG(file, sga.bestHist, sga.avgHist, file);
        std::cout << file << " done | best x = " << sga.bestX
            << " | fitness = " << sga.bestFit << "\n";
        };

    runSimpleExp(10, 0.7, 0.2, false, "exp1.svg");
    runSimpleExp(100, 0.7, 0.2, false, "exp2.svg");
    runSimpleExp(10, 0.1, 0.2, false, "exp3.svg");
    runSimpleExp(10, 0.7, 0.05, false, "exp4.svg");
    runSimpleExp(100, 0.1, 0.1, true, "exp5.svg");

    std::cout << "\nALL DONE\n";
    return 0;
}