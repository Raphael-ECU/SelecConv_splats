
import styletransfer_splat as styler

#hardcoded for testing
file_name = 'Table.ply'  #must be a .ply file
device = 'cpu'
style_path = 'style_ims\style0.jpg'
outPath ='Table_out_style0.splat'

styler.styletransfer(file_name,outPath,style_path, 25, device)