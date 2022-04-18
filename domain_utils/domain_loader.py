from torch.utils.data.dataloader import DataLoader



class DomainLoader(DataLoader):

    def __init__(self, dataset, domain_name, batch_sampler, num_workers, collate_fn):

        super().__init__(dataset=dataset,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=True)

        self.domain_name = domain_name
        self.dcp = f"Domain: {domain_name} | Size: {self.__len__()} | Batch Size {self.batch_size}"

    def __str__(self):
        return self.dcp
