#include<iostream>
#include<climits>     

// this method returns a minimum distance for the 
// vertex which is not included in Tset.
int minimumDist(int* dist, bool* Tset, int size) 
{
  int min=INT_MAX,index = -1;
              
  for(int i=0;i<size;i++) 
    {
      if(Tset[i]==false && dist[i]<=min)      
	{
	  min=dist[i];
	  index=i;
	}
    }
  return index;
}

void CPU_Dijkstra(int** graph,int src, int size, int* out=NULL) // adjacency matrix used is 6x6
{
  int dist[size]; // integer array to calculate minimum distance for each node.                            
  bool visisted[size];// boolean array to mark visted/unvisted for each node.
	
  // set the nodes with infinity distance
  // except for the initial node and mark
  // them unvisited.  
  for(int i = 0; i<size; i++) {
    dist[i] = INT_MAX;
    visisted[i] = false;	
  }
	
  dist[src] = 0;   // Source vertex distance is set to zero.
  
  int m = src;
  do {
    for(int i = 0; i<size; i++)                  
      {
	// Updating the minimum distance for the particular node.
	if(graph[m][i] != 0) {
	  if(!visisted[i]) {
	    if(dist[m]+graph[m][i] < dist[i]) {
	      dist[i]=dist[m]+graph[m][i];
	    }
	  }
	}
      }
    visisted[m]=true;// m with minimum distance included in visisted.
    m = minimumDist(dist,visisted, size); // vertex not yet included.
  } while(m != -1);

  if(out != NULL) {
    for(int i = 0; i < size; i++) {
      out[i] = dist[i];
    }
  }
}
