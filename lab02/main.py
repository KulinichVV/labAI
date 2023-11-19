import random
from deap import base, creator, tools

# Оценочная функция
def eval_func(individual):
    x, y, z = individual
    result = 1 / (1 + (x - 2)**2 + (y + 1)**2 + (z - 1)**2)
    return result,

# Создание инструментария с подходящими параметрами
def create_toolbox():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Инициализация инструментария
    toolbox = base.Toolbox()

    # Генерирование параметров x, y, z (атрибуты)
    toolbox.register("attr_float", random.uniform, -10, 10)

    # Инициализация структур
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, 3)

    # Определение популяции в виде списка индивидуумов
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Регистрация оператора оценки
    toolbox.register("evaluate", eval_func)

    # Регистрация оператора кроссовера
    toolbox.register("mate", tools.cxTwoPoint)

    # Регистрация оператора мутации
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # Оператор выбора индивидуумов для размножения
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

if __name__ == "__main__":
    # Создание набора инструментов
    toolbox = create_toolbox()

    # Затравочное значения для генератора случайных чисел
    random.seed(7)

    # Создание начальной популяции из 500 индивидуумов
    population = toolbox.population(n=500)

    # Определение вероятностей скрещивания и мутации
    probab_crossing, probab_mutating = 0.5, 0.2

    # Определение числа поколений
    num_generations = 30

    print('\n====Начало процесса эволюции====')

    # Проведение вычислений для всей популяции
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print('\nEvaluated', len(population), 'individuals')

    # Итерации по поколениям
    for g in range(num_generations):
        print("\n===== Generation", g)

        # Вывод текущих значений индивидов
        print("Current individual:\n", ind)

        # Выбор индивидов для перехода в следующее поколение
        offspring = toolbox.select(population, len(population))

        # Клонирование отобранных индивидов
        offspring = list(map(toolbox.clone, offspring))

        # Применение кроссовера и мутации к потомкам
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Скрестить двух индивидов
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)

                # "Забыть" параметры приспособленности детей
                del child1.fitness.values
                del child2.fitness.values

        # Применение мутации
        for mutant in offspring:
            # Мутация индивида
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Определить индивидов с недопустимыми значениями параметров приспособленности
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print('Evaluated', len(invalid_ind), 'individuals')

        # Популяция полностью заменяется потомками
        population[:] = offspring

    print("\n====Конец эволюции====")

    # Вывод окончательного результата
    best_ind = tools.selBest(population, 1)[0]
    print('\nЛучший индивид (x, y, z):\n', best_ind)
