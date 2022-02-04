from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    config = get_config(mode='train')
    test_config = get_config(mode='test')

    print(config)
    print(test_config)

    train_loader = get_loader(config.mode, config.video_type, config.split_index)
    test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index)
    solver = Solver(config, train_loader, test_loader)

    solver.build()
    solver.evaluate(-1)  #evaluates the set of selected thumbnails using the randomly initialized network
    solver.train()
