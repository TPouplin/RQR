best_parameter_dict = {
    "concrete":
    {
        "RQR-W": {
            "lr":0.0005,
            "scheduler1":0.999,
            "scheduler2":0.995,
            "penalty":0.1
        },
        
        "WS": {
            "lr":0.00001,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size":64,
            "epochs":1000
            },
        
        "QR": {
            "lr":0.00001,
            "scheduler1":0.999,
            "scheduler2":0.995
            },
        
        "IR": {
            "lr":0.001,
            "scheduler1":1,
            "scheduler2":1,
            "epochs":5000,
            "penalty":0.003,
            "batch_size": 512,
            },  
    },
    "boston":
    {
        "RQR-W": {
            "dropout":0.2,
            "penalty":0.1,
            "batch_size":512,
            "epochs":2000
        },
        
        "WS": {
            "dropout":0.1,
            "lr":0.00001,
            "batch_size": 64,
            "epochs":1000,
            
            },
        
        "QR": {
            "dropout":0.2           
            },
        
        "RQR-O": {
            "lr":1e-3,
            "dropout":0.,     
            "penalty":0.1,
            },
        
        "OQR": {       
            "lr":1e-3,
            "dropout":0.,     
            "penalty":0.1,
            },
        
        "IR": {
            "lr":0.001,
            "dropout":0.1,
            "penalty":0.1,
            "batch_size":512,
            "epochs":5000         
            },  
        
        "SQR": {
            "lr":1e-3,
            "dropout":0.,
            "penalty":0.,
            "batch_size":512,
            "epochs":5000         
            },  
    },
    "naval":
    {
        "RQR-W": {
            "lr":0.02,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size": 512,
            "epochs":2000,
            "penalty":15,
            
        },
        
        "WS": {
            "lr":0.000005,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size": 1024,
            "epochs":1000,      
            "dropout":0.1,
            },
        
        "QR": {
            "lr":0.02,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "batch_size": 512,
            "epochs":2000           
            },
        
        "IR": {
            "lr":0.005,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size": 1024,
            "epochs":2000,   
            "penalty":0.0005,
            "dropout":0.1        
            },  
    },
    "kin8nm":
    {
        "RQR-W": {
            "lr":0.02,
            "scheduler1":0.999,
            "scheduler2":0.998,
            "batch_size": 512,
            "penalty":5,
            "batch_size":1024,
            "epochs":3000
        },
        
        "WS": {
            "lr":0.00001,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size": 1024,           
            "epochs":1000,
            "dropout":0.1
            },
        
        "QR": {
            "lr":0.008,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "batch_size": 512,           
            },
        
        "IR": {
            "lr":0.005,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size": 1024, 
            "penalty":0.01,
            "epochs":2000     
            },  
    },
    "energy":
    {
        "RQR-W": {
            "lr":0.008,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "penalty":10
            
        },
        
        "WS": {
            "lr":0.00001,
            "scheduler1":1,
            "scheduler2":1,
            "epochs":1000
            
            },
        
        "QR": {
            "lr":0.008,
            "scheduler1":0.999,
            "scheduler2":0.999,
                       
            },
        
        "IR": {
            "lr":0.001,
            "scheduler1":1,
            "scheduler2":1,
            "penalty":0.002,
            "dropout":0,
            "epochs":3000,
            "batch_size": 512
                       
            },  
    },
    "power":
    {
        "RQR-W": {
            "lr":0.005,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "batch_size":512,
            "penalty":3
        },
        
        "WS": {
            "lr":0.005,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "batch_size":512,    
            
            },
        
        "QR": {
            "lr":0.00001,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size":1024,
            },
        
        "IR": {
            "lr":0.01,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size":1024,
            "penalty":0.001,
            "epochs":1000
            },  
    },
    "protein":
    {
        "RQR-W": {
            "lr":0.0005,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "batch_size":2048,
            "penalty":0.9
        },
        
        "WS": {
            "lr":0.00001,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size":1024,
            
            },
        
        "QR": {
            "lr":0.0005,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "batch_size":2048,
            },
        
        "IR": {
            "lr":0.005,
            "scheduler1":1,
            "scheduler2":1,
            "batch_size":2048,
            "penalty":0.1,
            "epochs":2000
            },  
    },
    "wine":
    {
        "RQR-W": {
            "lr":0.0005,
            "penalty": 1,
            "batch_size": 512,
            "epochs":2000,
            "dropout":0.3,
                        
        },
        
        "WS": {
            "lr":0.00001

            
            },
        
        "QR": {
            "lr":0.0001
            
            },
        
        "IR": {
            "lr":0.01,
            "penalty":0.1,
            "epochs":2000,
            "batch_size": 512
            },  
    },
    "yacht":
    {
        "RQR-W": {
            "lr":0.0008,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "epochs":2000,
            "penalty":0.9
        },
        
        "WS": {
            "lr":0.00005,
            "scheduler1":1,
            "scheduler2":1,
            "epochs":1000,
            
            },
        
        "QR": {
            "lr":0.001,
            "scheduler1":0.999,
            "scheduler2":0.999,
            "epochs":2000
            
            },
        
        "IR": {
            "lr":0.001,
            "scheduler1":1,
            "scheduler2":1,
            "epochs":2000,
            "penalty":0.1
            
            },  
    },
    
}