# import imageio
# images = []
# for filename in filenames:
    # images.append(imageio.imread(filename))
# imageio.mimsave('/path/to/movie.gif', images)

# import imageio
# with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
    # for filename in filenames:
        # image = imageio.imread(filename)
        # writer.append_data(image)
        
import imageio
import os.path as osp
import glob

figure_dir = "figures/gifs/graph"
movie_path = osp.join('./figures','gifs','hollow_1_graph'+'.gif')
geo_name = 'hollow_1'
# filenames = []
# for azim in range(0,360,10):
#     figure_path = "figures/gifs/" + geo_name+'_'+ str(azim) +"_graph_example.png"
#     filenames.append(figure_path)
filenames = glob.glob(osp.join(figure_dir, f'*.png'))
filenames.sort()
images = []
for filename in filenames:
    images.append(imageio.v2.imread(filename))
imageio.mimsave(movie_path, images)