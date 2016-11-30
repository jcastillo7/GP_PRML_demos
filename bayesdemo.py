import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def gaussian_process_demo(n=2,m=1000,fits=10):
    """
    Using a gaussian process we can evaluate the weights across all measurements
    but we cannot plot the most probable weights as they change for each location predicted.
    This is what gives the gaussian process its flexibility
    """
    #gaussian processes are a bit different compared to the counterpart used in bayesian learning
    #the basis function is a "radial basis function" which takes into account only training points, i.e. no
    #arbitrary interval is specified, and the number of data points is directly connected to the number of weights
    #this is an example of a non-parametric models
    return()


def demo_bayes_linear(n=2,m=1000,fits=10,alpha=2,beta=25,xtrain=None):
    """
    Linear demo of gaussian process... shown in PRML pg. 155
    """
    a0=-0.3
    a1=0.5
    #sample from function
    if xtrain==None:
        xtrain=np.random.uniform(-1,1,n)
    ytrain=a0+a1*xtrain

    #make design matrix
    dmat=np.vstack((np.ones((1,n)),xtrain.reshape(1,n)))
    PHImat=dmat.transpose()

    #Sampling From the Prior (prior is a zero mean gaussian)
    #alpha=2 #fixed
    #beta=np.square(1.0/0.2) # fixed
    c=np.eye(2)*1/alpha
    m0=np.array([0,0]).transpose()
    prior=np.random.multivariate_normal(m0,c,m)

    #Sampling From the posterior (posterior is a gaussian that estimates the weights of the function)
    Sninv=np.linalg.inv(c)+beta*np.dot(PHImat.transpose(),PHImat)
    mn=beta*np.dot(np.linalg.inv(Sninv),np.dot(np.linalg.inv(c),m0)+np.dot(PHImat.transpose(),ytrain))
    posterior=np.random.multivariate_normal(mn,np.linalg.inv(Sninv),m)

    #plotting test values
    test=np.arange(-1,1,0.1)
    testmat=np.vstack((np.ones(test.shape),test))
    ytest1=np.dot(posterior[np.random.randint(0,posterior.shape[0],fits),:],testmat)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    for y in ytest1:
        ax.plot(test,y)

    ax.scatter(xtrain,ytrain)
    fig.show()
    PHImat=dmat.transpose()

    fig2=plot_demo(prior,posterior)

    alpha,beta=evidence_func(PHImat,mn,ytrain,PHImat.transpose(),alpha,beta)
    print(np.mean(posterior,axis=0))
    return(xtrain,alpha,beta,fig,fig2)


def gaussian_basis(s=0.15,n=15,m=1000,fits=20,alpha=2,beta=7,itermax=5,Nbasis=9,plothist=False,demo="sin",xtrain=None):
    """
    This uses a gaussian basis function at fixed locations...
        i.e. 9 gaussian basis functions at equal intervals in the dataset

    example pg. 157

    Note:
    A direct extension of this regression process is Relevance Vector machines,
        -Relevance Vector machines allow the variance imposed of the prior distribution
            of the weights to not be fixed-->therefore learning these parameters
            results in an alpha for every weight (and results in sparse model as many weights
            are driven to zero pg. 345 PRML)
            --the changes are minor, but note while alpha is a matrix, beta is still a singular value

    """
    if demo=="marathon_demo":
        import pods
        data=pods.datasets.olympic_marathon_men()
        y=data["Y"].reshape(-1,) #size n
        if xtrain==None:
            xtrain=data["X"].reshape(-1,)#size n

        ytrain=y-y.mean()
        #makeup test set
        xtest=np.linspace(xtrain.min(),xtrain.max(),100) #size n*
        xtest.resize((100,1))
    else:
        #sample from function
        if xtrain==None:
            xtrain=np.random.uniform(-1,1,n)
        xtest=np.arange(-1,1,0.05)
        ytrain=np.sin(2*np.pi*xtrain)

    #use a non-linear basis function
    #in this case RBF=phi(x)=exp((x-mu_j)^2/(2s^2))
    if Nbasis == None:
        basis_location=xtrain
    else:
        basis_location=np.linspace(-1,1,Nbasis) #only use if want fixed location

    def setRBF(xtrain,s=0.45):
        def RBF_(x):
            dmat=np.zeros((x.shape[0],np.max(xtrain.shape[0])))
            for i in range(len(x)):
                for j in range(len(xtrain)):
                    dmat[i,j]=np.exp(-np.square(np.linalg.norm(x[i]-xtrain[j]))/(2*np.square(s)))
            return (dmat)
        return(RBF_)

    #if you want to use fixed locations for basis
    RBF_=setRBF(basis_location,s)
    n=len(basis_location)
    #make design matrix
    dmat=RBF_(xtrain)
    PHImat=dmat

    #Sampling From the Prior (prior is a zero mean gaussian)
    #beta=np.square(1.0/0.2) # fixed
    iter=0
    fign=plt.figure()
    ax1n=fign.add_subplot(121)
    ax2n=fign.add_subplot(122)
    while iter<itermax:
        c=np.eye(n)*1/alpha
        m0=np.zeros((n,))

        #Our prior is z zero mean gaussian
        prior=np.random.multivariate_normal(m0,np.linalg.inv(c),m)

        #Sampling From the posterior (posterior is a gaussian that estimates the weights of the function)
        Sninv=alpha*np.eye(n)+beta*np.dot(PHImat.transpose(),PHImat)
        mn=beta*np.dot(np.linalg.inv(Sninv),np.dot(PHImat.transpose(),ytrain))

        #posterior for the weights, i.e. sample all the weights.
        posterior=np.random.multivariate_normal(mn,np.linalg.inv(Sninv),m)

        alpha,beta=evidence_func(PHImat,mn,ytrain,RBF_(xtrain).transpose(),alpha,beta)
        #print("Evidence function suggests set alpha= %.2f" % np.real(alpha))
        #print("Evidence function suggests set beta= %.2f" % np.real(beta))
        ax1n.scatter(iter+1,alpha)
        ax2n.scatter(iter+1,beta)
        iter=iter+1

    #build testmat, i.e. solving for the testmatrix--> this is transforming all the inputs to gaussian
    #space and multiplying by the weights found in the previous step
    #the meaning of the weights are unclear in this context as they are weights of transformed data
    ax1n.set_ylabel("Alpha vs iter")
    ax2n.set_ylabel("Beta vs iter")
    fign.show()

    testmat=RBF_(xtest).transpose()
    ytest1=np.dot(posterior[np.random.randint(0,posterior.shape[0],fits),:],testmat)

    #build most likely function, i.e. expected value for the weights
    yall=np.dot(posterior,testmat)
    ymean=np.mean(yall,axis=0)

    #if demo=="marathon_demo":
    #    ymean=ymean+y.mean()
    #    ytest1=ytest1+y.mean()
    #print(wmean)
    #print(testmat.shape())
    #ymean=np.dot(wmean,testmat)


    fig=plt.figure()
    ax=fig.add_subplot(121)
    for y in ytest1:
        ax.plot(xtest,y)

    if demo=="sin":
        ax.set_ylim([-3,3])
        ax.set_xlim([-1,1])
    elif demo=="marathon_demo":
        ax.set_xlim([xtrain.min(),xtrain.max()])
        ax.set_ylim([ytrain.min()-1,ytrain.max()+1])
    ax.scatter(xtrain,ytrain)

    ax=fig.add_subplot(122)
    ax.plot(xtest,ymean,"b")
    if demo=="sin":
        ax.set_ylim([-3,3])
        ax.set_xlim([-1,1])
    elif demo=="marathon_demo":
        ax.set_xlim([xtrain.min(),xtrain.max()])
        ax.set_ylim([ytrain.min()-1,ytrain.max()+1])
    tmp=np.arange(-1,1,0.01)
    if demo=="sin":
        ax.plot(tmp,np.sin(2*np.pi*tmp),"r--")

    ax.scatter(xtrain,ytrain)

    fig.show()
    PHImat=dmat.transpose()

    if plothist==True:
        fig2=plot_demo_rbf(prior,posterior)

    #use evidence funciton to redetermine parameters maximizing likelihood of dataset
    #-- that step maximizes the likelihood of the training dataset p(t|alpha,beta)
    #effectively looking at p(alpha,beta|t)=p(t|alha,beta)p(alpha,beta)

    return(xtrain,alpha,beta)



def evidence_func(PHImat,mn,ytrain,xtrain_,alpha,beta):
    """
    Maximizing parameters based on training data only
    """

    X=beta*np.dot(PHImat.transpose(),PHImat)
    w,v=np.linalg.eig(X)
    lam=np.diag(w)
    gamma=np.sum(lam/(alpha+lam))
    alpha=gamma/(np.dot(mn.transpose(),mn))

    tmp=1/(xtrain_.shape[1]-gamma)*np.sum(np.square(ytrain-np.dot(mn,xtrain_)))
    beta=1/tmp

    return(alpha,beta)

def plot_demo_rbf(prior,posterior):
    #mu, sigma = 100, 15
    #x = mu + sigma*np.random.randn(10000)

    # the histogram of the data
    #n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
    fig=plt.figure()
    ax=fig.add_subplot(221)
    n, bins, patches = ax.hist(posterior[:,0],15,normed="True")
    plt.xlabel('Weights: w0')
    plt.ylabel('Probability')
    plt.grid(True)

    ax=fig.add_subplot(222)
    n, bins, patches = ax.hist(posterior[:,1],15,normed="True")
    # add a 'best fit' line
    #y = mlab.normpdf( bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Weights: w1')
    plt.ylabel('Probability')
    plt.grid(True)

    ax=fig.add_subplot(223)
    n, bins, patches = ax.hist(posterior[:,0],15,normed="True")
    plt.xlabel('Weights: w2')
    plt.ylabel('Probability')
    plt.grid(True)

    ax=fig.add_subplot(224)
    n, bins, patches = ax.hist(posterior[:,1],15,normed="True")
    # add a 'best fit' line
    #y = mlab.normpdf( bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Weights: w3')
    plt.ylabel('Probability')
    plt.grid(True)

    plt.show()
    return(fig)


def plot_demo(prior,posterior):
    #mu, sigma = 100, 15
    #x = mu + sigma*np.random.randn(10000)

    # the histogram of the data
    #n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
    fig=plt.figure()
    ax=fig.add_subplot(121)
    n, bins, patches = ax.hist(posterior[:,0],15,normed="True")
    plt.xlabel('Weights: w0')
    plt.ylabel('Probability')
    plt.grid(True)

    ax=fig.add_subplot(122)
    n, bins, patches = ax.hist(posterior[:,1],15,normed="True")
    # add a 'best fit' line
    #y = mlab.normpdf( bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Weights: w1')
    plt.ylabel('Probability')
    plt.grid(True)


    plt.show()
    return(fig)
