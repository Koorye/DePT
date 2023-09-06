# a tool for parallelizing running commands, where multiple commands are assigned to multiple graphics cards
import datetime
import os
from threading import Thread

from utils.logger import print


class RunCommandThread(Thread):
    def __init__(self, command):
        Thread.__init__(self)
        self.command = command

    def run(self):
        self.result = os.system(self.command)

    def get_result(self):
        return self.result


class GPUAllocater(object):
    def __init__(self, gpu_ids):
        self.gpu_ids = gpu_ids

        self.num_gpus = len(gpu_ids)
        self.commands = []
    
    def add_command(self, command):
        self.commands.append(command)
    
    def run(self):
        print('Summary of all commands:')
        for command in self.commands:
            command_ = command.replace('\\', '').replace('\n', ' ')
            print(command_[:75] + '...' + command_[-75:])
        print('=' * 40)
        
        current_command_idx, num_commands = 0, len(self.commands)
        print(f'Number of commands: {num_commands}\n')
        
        while len(self.commands) > 0:
            commands_once, self.commands = self.commands[:self.num_gpus], self.commands[self.num_gpus:]
            current_command_idx += len(commands_once)
            print(f'[{current_command_idx} / {num_commands}] Running commands:')
            self.run_once(commands_once)
    
    def run_once(self, commands):
        tasks = []

        print('=' * 40)
        for idx, command in enumerate(commands):
            gpu_id = self.gpu_ids[idx]
            command = f'CUDA_VISIBLE_DEVICES={gpu_id} ' + command
            
            print(command)
            if idx != len(commands) - 1:
                print('\n')
            
            t = RunCommandThread(command)
            tasks.append(t)

        print('=' * 40)
        print('Starting commands...')

        start_time = datetime.datetime.now()
        for t in tasks:
            t.start()
           
        for t in tasks:
            t.join()
        
        # raise exception when one of tasks does not run successfully
        results = [t.get_result() for t in tasks]
        for res in results:
            if res != 0:
                raise Exception('Commands cannot run properly!')

        end_time = datetime.datetime.now()
        print(f'Multi tasks FINISHED! Time cost: {end_time - start_time}\n')
