import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from torch import rand
import time
from os import popen
def check_mem():
    mem = popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split(
        ",")

    return mem


def memory_save():
    total, used = check_mem()

    total = int(total)
    used = int(used)

    max_mem = int(total * 0.5)
    block_mem = max_mem - used

    x = rand((256, 1024, block_mem),device="cuda")
    x = rand((2, 2),device="cuda")

    # do things here

if __name__ == '__main__':

    # torch.cuda.empty_cache()
    # torch.cuda.set_per_process_memory_fraction(1., 0)

    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    # memory_save()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        t = time.localtime()
        current_time = time.strftime("%D %H:%M:%S", t)
        print('start of epoch %d / %d at %s' %
              (epoch, opt.niter + opt.niter_decay,current_time))
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
        t = time.localtime()
        current_time = time.strftime("%D %H:%M:%S", t)
        print('End of epoch %d / %d \t Time Taken: %d sec \t current time: %s' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time,current_time))
        model.update_learning_rate()
